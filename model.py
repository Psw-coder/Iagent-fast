# This is a main script that tests the functionality of specific agents.
# It requires no user input.
"""
中文说明：
本文件实现两个基于大模型的推荐重排代理 iAgent 与 i2Agent。
- iAgent：两阶段流程，先抽取推荐知识，再对候选集重排。
- i2Agent：在 iAgent 基础上增加用户画像与动态兴趣抽取，再进行重排。
最终统一输出 HIT/NDCG/MRR 评估指标。
"""


import json
import pandas as pd
import os
import warnings
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import re
from openai import OpenAI
import openai
import logging
from pathlib import Path
import time
import random

def cal_ndcg_hr_single(answer,ranking_list,topk=10):
    """
    计算单条样本在给定 topk 下的排序指标。
    参数：
    - answer：真实目标 item id
    - ranking_list：模型输出的重排列表
    - topk：评估截断位置
    返回：
    - HIT：命中返回 1，否则 0；若 answer 不在列表中返回 -1
    - NDCG：命中时按排名衰减计算；异常时为 -1
    - MRR：倒数排名；异常时为 -1
    """
    try:
        rank = ranking_list.index(answer)
        # print(rank)
        HIT = 0
        NDCG = 0
        MRR = 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG = 1.0 / np.log2(rank + 2.0)
            HIT = 1.0
    except ValueError:
        HIT = -1 
        NDCG = -1
        MRR = -1
    return HIT , NDCG , MRR 

class iAgent():
    """
    iAgent 两阶段代理。
    阶段1：根据任务指令生成知识描述（不直接推荐具体商品）。
    阶段2：结合候选列表、知识、静态兴趣，生成 rerank_list 和解释。
    """

    def __init__(self, task_input, logger):
        self.task_input = task_input
        self.messages = []
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        self.workflow = [
            {
                "message": "Based on the following instruction, assist me in generating relevant knowledge. Please specify the types of descriptions that the recommended items should include. Do not directly recommend specific items. ",
                "tool_use": None
            },
            {
                "message": "Based on the information, give recommendations for the user based on the constraints. ",
                "tool_use": None
            }
        ]
        self.logger = logger

    def run(self):
        """
        主执行流程（iAgent）：
        1) 从 task_input 读取指令、用户历史、候选集、答案等字段。
        2) 构建 user_memory（静态兴趣摘要）。
        3) 按 workflow 依次调用模型：
           - 第一步抽取 knowledge
           - 第二步输出 rerank_list 与 explanation
        4) 计算 HIT@1/3/5、NDCG@1/3/5、MRR。
        5) 若输出异常（MRR=-1），触发最多 3 次纠错重试。
        """
        task_input = self.task_input
        instruction,title,description, asin,answer,candidate_ranked_list,pure_ranked_list = task_input['instruction'],task_input['title'],task_input['description'],task_input['asin'],task_input['answer'],task_input['ranked_list_str'],task_input['pure_ranked_list']
        reviewText = task_input["reviewText"]
        
        user_memory = ""
        for j in range(len(asin)):
            # Ensure the description is properly handled as a string
            description_str = description[j][-200:] if isinstance(description[j][-200:], str) else str(description[j][-200:])
            user_memory += "user historical information, item title:{},item description:{} ;".format(
                title[j], re.sub(u"\\<.*?\\>", "", description_str)
            )

        user_memory_previous = ""
        for j in range(len(asin)-1):
            # Ensure the description and reviewText are properly handled as strings
            description_str = description[j][-200:] if isinstance(description[j][-200:], str) else str(description[j][-200:])
            review_str = reviewText[j][-200:] if isinstance(reviewText[j][-200:], str) else str(reviewText[j][-200:])
            user_memory_previous += "title:{},description:{},review:{} \t ".format(
                title[j], re.sub(u"\\<.*?\\>", "", description_str),
                re.sub(u"\\<.*?\\>", "", review_str)
            )

        workflow = self.workflow

        try:
            if workflow:
                MRR = None
                for i, step in enumerate(workflow):
                    message = step["message"]
                    if i == 0:
                        self.messages.append({
                            "role": "assistant",
                            "content": "{} \n. Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Instruction:{}".format(message,instruction)
                            })
                        retries = 0
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                                messages=self.messages,
                                                model= "gpt-4o-mini",
                                                response_format={
                                                    "type": "json_schema",
                                                    "json_schema": {
                                                        "name": "custom_response",
                                                        "schema": {
                                                            "type": "object",
                                                            "properties": {
                                                                "knowledge": {
                                                                    "type": "string",
                                                                },
                                                            },
                                                            "required": ["knowledge"],
                                                            "additionalProperties": False
                                                        },
                                                        "strict": True
                                                    }
                                                }
                                            )
                                try:
                                    knowledge_tool_str = json.loads(completion.choices[0].message.content)["knowledge"]
                                    break
                                except Exception as e:
                                    self.logger.info(f"An unexpected error occurred: {e}")
                                    knowledge_tool_str = "Extract Error!"
                                    retries += 1

                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                knowledge_tool_str = "Extract Error!"
                                retries += 1
                                self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                                time.sleep(5)  


                        str_instruction_print = "{} \n. Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Instruction:{}".format(message,instruction)
                        self.logger.info(f"based on the message:{str_instruction_print}, LLM generate knowledge is: {knowledge_tool_str}\n")

                    if i == len(workflow) - 1:
                        self.messages.append({
                            "role": "assistant",
                            "content": "{}.\n Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Candidate ranking list:{},Knowledge:{},Static Interest:{}, Pure Ranking List:{}".format(message,candidate_ranked_list,knowledge_tool_str,user_memory,pure_ranked_list)
                            })
                        retries = 0
                        ranker_str_logger = "{}.\n Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Candidate ranking list:{},Knowledge:{},Static Interest:{}, Pure Ranking List:{}".format(message,candidate_ranked_list,knowledge_tool_str,user_memory,pure_ranked_list)
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                    messages=self.messages,
                                    model= "gpt-4o-mini",
                                    response_format={
                                        "type": "json_schema",
                                        "json_schema": {
                                            "name": "custom_response",
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "rerank_list": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "integer"
                                                        }
                                                    },
                                                    "explanation": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "string"
                                                        }
                                                    }
                                                },
                                                "required": ["rerank_list", "explanation"],
                                                "additionalProperties": False
                                            },
                                            "strict": True
                                        }
                                    }
                                )
                                response = completion.choices[0].message.content
                                try:
                                    response_dict = json.loads(response)
                                    rerank_list,explanation = response_dict["rerank_list"],response_dict["explanation"]
                                    break
                                except Exception as e:
                                    self.logger.info(f"An unexpected error occurred: {e}")
                                    rerank_list = "Extract Error!"
                                    explanation = "Extract Error!"
                                    retries += 1
                            
                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                rerank_list = "Extract Error!"
                                explanation = "Extract Error!"
                                retries += 1
                                self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                                time.sleep(5)
                                # break


                        self.logger.info("message:{},explanation:{},pure_ranking_list:{}, answer, {} llm_ranking_list, {}\n".format(ranker_str_logger,explanation,pure_ranked_list,answer,rerank_list))
                        HIT_1 , NDCG_1 , MRR  = cal_ndcg_hr_single(answer,rerank_list,1)
                        HIT_3 , NDCG_3 , MRR  = cal_ndcg_hr_single(answer,rerank_list,3)
                        HIT_5 , NDCG_5 , MRR  = cal_ndcg_hr_single(answer,rerank_list,5)

                retry_mrr_times = 0
                while MRR == -1 and retry_mrr_times<3:
                    retry_mrr_times += 1
                    self.logger.info(f"Generate error ranking list: {rerank_list}\n")
                    retry_message_str = "Rerank list is out of the order, you should rerank the item from the pure ranking list. The previous list:{}. Therefore, try it again according the following information.".format(rerank_list)
                    self.messages.append({
                            "role": "assistant",
                            "content": "{}. \n Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Candidate ranking list:{},Knowledge:{},,Static Inster:{},  Please generate the reranked list from Pure Ranking List:{}. The length of the reranked list should be 10.".format(retry_message_str,candidate_ranked_list,knowledge_tool_str,user_memory,pure_ranked_list)
                            })
                    retries = 0
                    while retries < 3:
                        try:
                            completion = self.client.chat.completions.create(
                                messages=self.messages,
                                model= "gpt-4o-mini",
                                response_format={
                                    "type": "json_schema",
                                    "json_schema": {
                                        "name": "custom_response",
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "rerank_list": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "integer"
                                                    }
                                                },
                                                "explanation": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            },
                                            "required": ["rerank_list", "explanation"],
                                            "additionalProperties": False
                                        },
                                        "strict": True
                                    }
                                }
                            )
                            response = completion.choices[0].message.content
                            try:
                                response_dict = json.loads(response)
                                rerank_list,explanation = response_dict["rerank_list"],response_dict["explanation"]
                                break
                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                rerank_list = "Extract Error!"
                                explanation = "Extract Error!"
                                retries += 1
                        
                        except Exception as e:
                            self.logger.info(f"An unexpected error occurred: {e}")
                            rerank_list = "Extract Error!"
                            retries += 1
                            self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                            time.sleep(5)
                            # break
                    
                    self.logger.info("one more time, pure_ranking_list:{}, answer, {} llm_ranking_list, {}\n".format(pure_ranked_list,answer,rerank_list))
                    HIT_1 , NDCG_1 , MRR  = cal_ndcg_hr_single(answer,rerank_list,1)
                    HIT_3 , NDCG_3 , MRR  = cal_ndcg_hr_single(answer,rerank_list,3)
                    HIT_5 , NDCG_5 , MRR  = cal_ndcg_hr_single(answer,rerank_list,5)

                return {
                    "HIT":(HIT_1,HIT_3,HIT_5),
                    "NDCG":(NDCG_1,NDCG_3,NDCG_5),
                    "MRR":MRR,
                }

            else:
                return {
                    "HIT":(-1,-1,-1),
                    "NDCG":(-1,-1,-1),
                    "MRR":-1,
                }
                    
        except Exception as e:
            self.logger.error(e)
            return {
                    "HIT":(-1,-1,-1),
                    "NDCG":(-1,-1,-1),
                    "MRR":-1,
                }        

class i2Agent():
    """
    i2Agent 增强代理。
    相比 iAgent，新增“用户画像 + 动态兴趣”阶段，使重排依据更丰富。
    """

    def __init__(self,task_input,logger):
        self.task_input = task_input
        self.messages = []
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.workflow = [
            {
                "message": "Here is the background of one user. ",
                "tool_use": None
            },
            {
                "message": "",
                "tool_use": None
            },
            {
                "message": "Based on the following instruction, assist me in generating relevant knowledge. Please specify the types of descriptions that the recommended items should include. Do not directly recommend specific items. ",
                "tool_use": None
            },
            {
                "message": "Based on the generated knowledge and the instruction, extract some dynamic interest information from the static memory. Moreover, based on the profile and the instruction, extract some dynamic profile information. ",
                "tool_use": None
            },
            {
                "message": "Based on the information, give recommendations for the user based on the constrains. ",
                "tool_use": None
            }
        ]
        self.logger = logger

    def run(self):
        """
        主执行流程（i2Agent）：
        1) 截取最近行为，构建静态兴趣 user_memory。
        2) 多阶段调用模型：
           - Step0：正负样本对比推荐（构造偏好背景）
           - Step1：根据真实选择与评论抽取 profile
           - Step2：根据 instruction 生成 knowledge
           - Step3：生成 dynamic_interest 与 dynamic_profile
           - Step4：融合静态/动态信息完成最终重排
        3) 计算 HIT/NDCG/MRR，并在异常时执行重试纠错。
        """
        max_length = 15
        task_input = self.task_input
        instruction,title,description, asin,answer,candidate_ranked_list,pure_ranked_list = task_input['instruction'],task_input['title'],task_input['description'],task_input['asin'],task_input['answer'],task_input['ranked_list_str'],task_input['pure_ranked_list']
        reviewText,neg_sample_title,neg_sample_descript = task_input["reviewText"],task_input["neg_sample_title"],task_input["neg_sample_descript"]
        title,description, asin = title[-max_length:],description[-max_length:], asin[-max_length:]

        user_memory = ""
        for j in range(len(asin)):
            # Ensure the description is properly handled as a string
            description_str = description[j][-200:] if isinstance(description[j][-200:], str) else str(description[j][-200:])
            user_memory += "user historical information, item title:{},item description:{} ;".format(
                title[j], re.sub(u"\\<.*?\\>", "", description_str)
            )

        user_memory_previous = ""
        for j in range(len(asin)-1):
            # Ensure the description and reviewText are properly handled as strings
            description_str = description[j][-200:] if isinstance(description[j][-200:], str) else str(description[j][-200:])
            review_str = reviewText[j][-200:] if isinstance(reviewText[j][-200:], str) else str(reviewText[j][-200:])
            user_memory_previous += "title:{},description:{},review:{} \t ".format(
                title[j], re.sub(u"\\<.*?\\>", "", description_str),
                re.sub(u"\\<.*?\\>", "", review_str)
            )
        workflow = self.workflow

        try:
            self.messages_initial = []
            self.messages_neighbor = []
            if workflow:
                MRR = None
                for i, step in enumerate(workflow):
                    message = step["message"]
                    tool_use = step["tool_use"]

                    if i == 0:
                        step_one_message_str = "{} \n. Please recommend one item for her. The first one title:{}, descrition:{}. The second one title:{}, description:{}. ".format(message,title[-2],re.sub(u"\\<.*?\\>", "",str(description[-2][-200:])),neg_sample_title,re.sub(u"\\<.*?\\>", "",str(neg_sample_descript[-200:])))
                        self.messages_initial.append({
                            "role": "assistant",
                            "content": "{} \n. Please recommend one item for her. The first one title:{}, descrition:{}. The second one title:{}, description:{}. ".format(message,title[-2],re.sub(u"\\<.*?\\>", "",str(description[-2][-200:])),neg_sample_title,re.sub(u"\\<.*?\\>", "",str(neg_sample_descript[-200:])))
                            })
                        retries = 0
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                                messages=self.messages_initial,
                                                model= "gpt-4o-mini",
                                                response_format={
                                                    "type": "json_schema",
                                                    "json_schema": {
                                                        "name": "custom_response",
                                                        "schema": {
                                                            "type": "object",
                                                            "properties": {
                                                                "recommend_content": {
                                                                    "type": "string",
                                                                },
                                                            },
                                                            "required": ["recommend_content"],
                                                            "additionalProperties": False
                                                        },
                                                        "strict": True
                                                    }
                                                }
                                            )
                                try:
                                    recommend_content = json.loads(completion.choices[0].message.content)["recommend_content"]
                                    self.messages_initial.append({
                                        "role": "assistant",
                                        "content": "The recommend content: {} \n. ".format(recommend_content)
                                        })
                                    break
                                except Exception as e:
                                    self.logger.info(f"An unexpected error occurred: {e}")
                                    recommend_content = "Extract Error!"
                                    retries += 1    
                            
                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                recommend_content = "Extract Error!"
                                retries += 1
                                self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                                time.sleep(5)
                                # break

                        self.logger.info("The first step message:{} \n".format(step_one_message_str))
                        self.logger.info(f"generate recommend_content is: {recommend_content}\n")

                    if i == 1:
                        second_message_str = "{} \n. Great! Actually, this user choose the item with title:{} and review:{}. Can you generate the profile of this user background? Please make a detailed profile. Don’t use numerical numbering for the generated content; you can use bullet points instead.".format(message,title[-2],reviewText[-2][-200:])
                        self.messages_initial.append({
                            "role": "assistant",
                            "content": "{} \n. Great! Actually, this user choose the item with title:{} and review:{}. Can you generate the profile of this user background? Please make a detailed profile. Don’t use numerical numbering for the generated content; you can use bullet points instead.".format(message,title[-2],reviewText[-2][-200:])
                            })
                        retries = 0
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                                messages=self.messages_initial,
                                                model= "gpt-4o-mini",
                                                response_format={
                                                    "type": "json_schema",
                                                    "json_schema": {
                                                        "name": "custom_response",
                                                        "schema": {
                                                            "type": "object",
                                                            "properties": {
                                                                "profile": {
                                                                    "type": "string",
                                                                },
                                                            },
                                                            "required": ["profile"],
                                                            "additionalProperties": False
                                                        },
                                                        "strict": True
                                                    }
                                                }
                                            )
                                try:
                                    profile = json.loads(completion.choices[0].message.content)["profile"]
                                    break
                                except Exception as e:
                                    self.logger.info(f"An unexpected error occurred: {e}")
                                    profile = "Extract Error!"
                                    retries += 1
                            
                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                profile = "Extract Error!"
                                retries += 1
                                self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                                time.sleep(5) 
                                # break

                        self.logger.info("second message :{}\n".format(second_message_str))
                        self.logger.info(f"generate profile is: {profile}\n")

                    if i == 2:
                        self.messages.append({
                            "role": "assistant",
                            "content": "{} \n. Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Instruction:{}".format(message,instruction)
                            })
                        retries = 0
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                                messages=self.messages,
                                                model= "gpt-4o-mini",
                                                response_format={
                                                    "type": "json_schema",
                                                    "json_schema": {
                                                        "name": "custom_response",
                                                        "schema": {
                                                            "type": "object",
                                                            "properties": {
                                                                "knowledge": {
                                                                    "type": "string",
                                                                },
                                                            },
                                                            "required": ["knowledge"],
                                                            "additionalProperties": False
                                                        },
                                                        "strict": True
                                                    }
                                                }
                                            )
                                try:
                                    knowledge_tool_str = json.loads(completion.choices[0].message.content)["knowledge"]
                                    break
                                except Exception as e:
                                    self.logger.info(f"An unexpected error occurred: {e}")
                                    knowledge_tool_str = "Extract Error!"
                                    retries += 1
                            
                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                knowledge_tool_str = "Extract Error!"
                                retries += 1
                                self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                                time.sleep(5)  
                                # break

                        
                        self.logger.info(f"generate knowledge is: {knowledge_tool_str}\n")

                    if i == 3:
                        forth_message_str = "{}. Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Generated Knowledge:{} Instruction:{} Historical Information:{} Profile:{}".format(message,knowledge_tool_str,instruction,user_memory,profile)
                        self.messages.append({
                            "role": "assistant",
                            "content": "{}. Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Generated Knowledge:{} Instruction:{} Historical Information:{} Profile:{}".format(message,knowledge_tool_str,instruction,user_memory,profile)
                            })
                        retries = 0
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                    messages=self.messages,
                                    model= "gpt-4o-mini",
                                    response_format={
                                        "type": "json_schema",
                                        "json_schema": {
                                            "name": "custom_response",
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "dynamic_interest": {
                                                        "type": "string",
                                                    },
                                                    "dynamic_profile": {
                                                        "type": "string",
                                                    },
                                                },
                                                "required": ["dynamic_interest","dynamic_profile"],
                                                "additionalProperties": False
                                            },
                                            "strict": True
                                        }
                                    }
                                )
                                try:
                                    dynamic_interest_str = json.loads(completion.choices[0].message.content)["dynamic_interest"]
                                    dynamic_profile_str = json.loads(completion.choices[0].message.content)["dynamic_profile"]
                                    break
                                except Exception as e:
                                    self.logger.info(f"An unexpected error occurred: {e}")
                                    dynamic_interest_str = "Extract Error!"
                                    dynamic_profile_str = "Extract Error!"
                                    retries += 1
                            
                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                dynamic_interest_str = "Extract Error!"
                                retries += 1
                                self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                                time.sleep(5)  

                        self.logger.info("forth step message is :{}\n".format(forth_message_str))
                        self.logger.info(f"dynamic interest is: {dynamic_interest_str} dynamic_profile_str:{dynamic_profile_str}\n")


                    if i == len(workflow) - 1:
                        self.messages.append({
                            "role": "assistant",
                            "content": "{}.\n Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Candidate ranking list:{},Knowledge:{},Dynamic Interest:{},Static Interest:{}, Static User Profile:{}, Dynamic User Profile:{}, Pure Ranking List:{}".format(message,candidate_ranked_list,knowledge_tool_str,dynamic_interest_str,user_memory,profile,dynamic_profile_str,pure_ranked_list)
                            })
                        retries = 0
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                    messages=self.messages,
                                    model= "gpt-4o-mini",
                                    response_format={
                                        "type": "json_schema",
                                        "json_schema": {
                                            "name": "custom_response",
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "rerank_list": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "integer"
                                                        }
                                                    },
                                                    "explanation": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "string"
                                                        }
                                                    }
                                                },
                                                "required": ["rerank_list", "explanation"],
                                                "additionalProperties": False
                                            },
                                            "strict": True
                                        }
                                    }
                                )
                                response = completion.choices[0].message.content
                                try:
                                    response_dict = json.loads(response)
                                    rerank_list,explanation = response_dict["rerank_list"],response_dict["explanation"]
                                    break
                                except Exception as e:
                                    self.logger.info(f"An unexpected error occurred: {e}")
                                    rerank_list = "Extract Error!"
                                    explanation = "Extract Error!"
                                    retries += 1
                            
                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                rerank_list = "Extract Error!"
                                explanation = "Extract Error!"
                                retries += 1
                                self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                                time.sleep(5) 
                                # break

                        self.logger.info("pure_ranking_list:{}, answer, {} llm_ranking_list, {}\n".format(pure_ranked_list,answer,rerank_list))
                        HIT_1 , NDCG_1 , MRR  = cal_ndcg_hr_single(answer,rerank_list,1)
                        HIT_3 , NDCG_3 , MRR  = cal_ndcg_hr_single(answer,rerank_list,3)
                        HIT_5 , NDCG_5 , MRR  = cal_ndcg_hr_single(answer,rerank_list,5)

                retry_mrr_times = 0
                while MRR == -1 and retry_mrr_times<3:

                    retry_mrr_times += 1
                    self.logger.info(f"Generate error ranking list: {rerank_list}\n")
                    retry_message_str = "Rerank list is out of the order, you should rerank the item from the pure ranking list. The previous list:{}. Therefore, try it again according the following information.".format(rerank_list)
                    self.messages.append({
                            "role": "assistant",
                            "content": "{}. \n Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Candidate ranking list:{},Knowledge:{},Dynamic Interest:{},Static Inster:{}, User Profile:{}, Please generate the reranked list from Pure Ranking List:{}. The length of the reranked list should be 10.".format(retry_message_str,candidate_ranked_list,knowledge_tool_str,dynamic_interest_str,user_memory,profile,pure_ranked_list)
                            })
                    retries = 0
                    while retries < 3:
                        try:
                            # 尝试发送请求
                            completion = self.client.chat.completions.create(
                                messages=self.messages,
                                model= "gpt-4o-mini",
                                response_format={
                                    "type": "json_schema",
                                    "json_schema": {
                                        "name": "custom_response",
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "rerank_list": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "integer"
                                                    }
                                                },
                                                "explanation": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            },
                                            "required": ["rerank_list", "explanation"],
                                            "additionalProperties": False
                                        },
                                        "strict": True
                                    }
                                }
                            )
                            response = completion.choices[0].message.content
                            try:
                                response_dict = json.loads(response)
                                rerank_list,explanation = response_dict["rerank_list"],response_dict["explanation"]
                                break
                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                rerank_list = "Extract Error!"
                                explanation = "Extract Error!"
                                retries += 1
                        
                        except Exception as e:
                            self.logger.info(f"An unexpected error occurred: {e}")
                            rerank_list = "Extract Error!"
                            retries += 1
                            self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                            time.sleep(5)
                            # break
                    
                    self.logger.info("one more time, pure_ranking_list:{}, answer, {} llm_ranking_list, {}\n".format(pure_ranked_list,answer,rerank_list))
                    HIT_1 , NDCG_1 , MRR  = cal_ndcg_hr_single(answer,rerank_list,1)
                    HIT_3 , NDCG_3 , MRR  = cal_ndcg_hr_single(answer,rerank_list,3)
                    HIT_5 , NDCG_5 , MRR  = cal_ndcg_hr_single(answer,rerank_list,5)

                return {
                    "HIT":(HIT_1,HIT_3,HIT_5),
                    "NDCG":(NDCG_1,NDCG_3,NDCG_5),
                    "MRR":MRR,
                }

            else:
                return {
                    "HIT":(-1,-1,-1),
                    "NDCG":(-1,-1,-1),
                    "MRR":-1,
                }
                    
        except Exception as e:
            self.logger.error(e)
            return {
                    "HIT":(-1,-1,-1),
                    "NDCG":(-1,-1,-1),
                    "MRR":-1,
                }
