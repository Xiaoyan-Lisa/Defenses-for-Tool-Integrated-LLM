import os
import json
from .agent_process import (
    AgentProcess
)
import time
from threading import Thread
from ..utils.logger import AgentLogger
from ..utils.chat_template import Query
import importlib
from ..queues.llm_request_queue import LLMRequestQueue
from pyopenagi.tools.simulated_tool import SimulatedTool
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from openai import OpenAI, AzureOpenAI
import copy

class CustomizedThread(Thread):
    def __init__(self, target, args=()):
        super().__init__()
        self.target = target
        self.args = args
        self.result = None

    def run(self):
        self.result = self.target(*self.args)

    def join(self):
        super().join()
        return self.result

class BaseAgent:
    def __init__(self,
                 agent_name,
                 task_input,
                 agent_process_factory,
                 log_mode: str
        ):
        self.agent_name = agent_name
        self.config = self.load_config()
        self.tool_names = self.config["tools"]
        self.agent_process_factory = agent_process_factory
        self.tool_list = dict()
        self.tools = []
        self.num_tools = []
        self.start_time = None
        self.end_time = None
        self.request_waiting_times: list = []
        self.request_turnaround_times: list = []
        self.task_input = task_input
        self.messages = []
        self.workflow_mode = "manual" # (mannual, automatic)
        self.rounds = 0
        self.log_mode = log_mode
        self.logger = self.setup_logger()
        self.set_status("active")
        self.set_created_time(time.time())

    def run(self):
        '''Execute each step to finish the task.'''
        pass

    # can be customization
    def build_system_instruction(self):
        pass

    # def check_workflow_updated(self, message):
    #     try:
    #         workflow = json.loads(message)
    #         if not isinstance(workflow, list):
    #             workflow = [workflow]
    #         if len(workflow)>= 2:
    #             for step in workflow:
    #                 if "message" not in step or "tool_use" not in step:
    #                     return None
    #             return workflow
    #         else:
    #             return None
    #     except json.JSONDecodeError:
    #         return None
        
    def check_workflow(self, message):
        try:
            workflow = json.loads(message)
            if not isinstance(workflow, list):
                workflow = [workflow]
            for step in workflow:
                if "message" not in step or "tool_use" not in step:
                    return None
            return workflow
        except json.JSONDecodeError:
            return None
        
    def evaluator(self, workflow):
        client = OpenAI(api_key=OPENAI_API_KEY)
        system_prompt = f"""You are a helpful assistant. Your goal is to check the validity of the given workflow.
                    The workflow is formatted as a JSON list and must contain at least two steps, such as:
                    [
                        {{
                            "message": "Use the MarketDataAPI tool to gather recent market data and trends.",
                            "tool_use": ["MarketDataAPI"]
                        }},
                        {{
                            "message": "Postprocess the gathered data to provide investment recommendations.",
                            "tool_use": ["PortfolioManager"]
                        }}
                    ]
                    Each item in the list is a dictionary with:
                    - 'message': a description of the agent's action
                    - 'tool_use': a list of tools used for this step
                    
                    The available tools are:
                    {json.dumps(self.tools, indent=2)}
                    
                    Your job is to:
                    - Check that the workflow uses only tools from the list above
                    - Ensure the workflow has at least two steps
                    - Make sure each step logically follows the previous one in a reasonable sequence
                    
                    If the workflow is valid and meets all of these criteria, return it exactly as it is.
                    If there are problems (e.g., tool mismatch, missing steps, or illogical step ordering), return 'None' and briefly explain what needs to be improved.
                    """
                    
                                
        try:
            workflow_str = json.dumps(workflow, indent=2)
    
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": workflow_str}
                ]
            )
            result = completion.choices[0].message.content.strip()
            if result.lower().startswith("none"):
                print("[Reflection] Rejected workflow.")
                reason = result if result.lower() != "none" else "No reason provided."
                return None, reason
            else:
                print(f"[Reflection] Accepted workflow:\n{result}")
                return result, None
        except Exception as e:
            print(f"[Reflection Error] Skipping feedback. Reason: Error: {str(e)}")
            return None, f"Error: {str(e)}"
    
    def self_reflection(self, workflow, max_attempts=5):
        original_workflow = workflow  # backup
        self.messages.append(
            {
                "role": "system",
                "content": (
                    f"Please validate the following agent workflow. The workflow is formatted as a JSON list.\n\n"
                    f"Workflow:\n{json.dumps(workflow, indent=2)}\n\n"
        
                    "Each step in the workflow must be a dictionary with the following keys:\n"
                    "- 'message': a description of what the agent does in that step.\n"
                    "- 'tool_use': a list of tool names used in that step.\n\n"
        
                    f"From now on, you are only allowed to use the normal tools listed below:\n{json.dumps(self.tools, indent=2)}\n\n"
        
                    "Validation rules:\n"
                    "- The workflow must contain at least **two steps**.\n"
                    "- Each step must include a **non-empty** 'tool_use' field.\n"
                    "- All tools in 'tool_use' must be selected **only from the allowed tools list above**.\n"
                    "- Steps must follow a **logical sequence** — each step should build on the previous one.\n"
                    "- Do **not** modify the 'message' field unless it is incorrect or unclear.\n"
                    "- Steps should **not repeat** unnecessarily.\n\n"
        
                    "If the workflow is valid, return it exactly as it is.\n"
                    "If the workflow has issues, return a corrected version in the same format.\n"
                    "Respond ONLY with the valid or corrected JSON list. Do not include any extra explanation or text."
                )
            }
        )

        for attempt in range(max_attempts):
            try:
                response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
                    query=Query(
                        messages=self.messages,  ##  messages
                        tools=None,
                        message_return_type="json"
                    )
                )
                if attempt == 0:
                    self.set_start_time(start_times[0])
    
                self.request_waiting_times.extend(waiting_times)
                self.request_turnaround_times.extend(turnaround_times)
                if self.args.direct_prompt_injection:
                    corrected_workflow = self.check_workflow(response.response_message)
                else:
                    corrected_workflow = self.check_workflow_updated(response.response_message)

                if corrected_workflow:
                    print(f"[Reflection ✅] Valid workflow returned at attempt {attempt + 1}: {corrected_workflow} ")
                    return corrected_workflow
                self.messages.append({
                    "role": "user",
                    "content": "The previous attempt was invalid. Please try again to generate a valid workflow."
                })

            except Exception as e:
                print(f"[Reflection ❌] Attempt {attempt + 1} failed due to error:\n{e}")
    
        print("[Reflection ⚠️] All attempts failed. Returning original workflow.")
        return original_workflow 

    def automatic_workflow(self):
        for i in range(self.plan_max_fail_times):
            response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
                query=Query(
                    messages= self.messages,
                    tools=None,
                    message_return_type="json"
                )
            )
            if self.rounds == 0:
                self.set_start_time(start_times[0])
    
            self.request_waiting_times.extend(waiting_times)
            self.request_turnaround_times.extend(turnaround_times)
            workflow = self.check_workflow(response.response_message)
    
            self.rounds += 1
            if workflow:
                if self.args.reflection:
                    workflow_ = self.self_reflection(workflow)
                    # workflow_reflect = self.check_workflow(workflow_)
                    if workflow_:
                        print(f"[Reflected_workflow ✅] Return reflected workflow: {workflow_} ")
                        return workflow_
                    else:
                        return workflow
                else:
                    return workflow
         
            if self.args.llm_name == 'claude-3-5-sonnet-20240620':
                self.messages.append({
                    "role": "assistant",
                    "content": f"Fail {i+1} times to generate a valid plan. I need to regenerate a plan."
                })
                self.messages.append({
                    "role": "user",
                    "content": f"Please try again. Fail {i+1} times to generate a valid plan. I need to regenerate a plan."
                })
            else:
                self.messages.append({
                    "role": "assistant",
                    "content": f"Fail {i+1} times to generate a valid plan. I need to regenerate a plan."
                })
    
            if i == self.plan_max_fail_times - 1:
                self.messages.append({
                    "role": "assistant",
                    "content": f"[Thinking]: {response.response_message}"
                })
                print("[Max Retry Reached] No valid workflow. Returning None.")
                return None
    
        return None
        
    
    def manual_workflow(self):
        pass

    def snake_to_camel(self, snake_str):
        components = snake_str.split('_')
        return ''.join(x.title() for x in components)

    def load_tools(self, tool_names):
        for tool_name in tool_names:
            org, name = tool_name.split("/")
            module_name = ".".join(["pyopenagi", "tools", org, name])
            class_name = self.snake_to_camel(name)

            tool_module = importlib.import_module(module_name)
            tool_class = getattr(tool_module, class_name)

            self.tool_list[name] = tool_class()
            self.tools.append(tool_class().get_tool_call_format())
            
    def load_tools_from_file(self, tool_names, tools_info):
        for tool_name in tool_names:
            org, name = tool_name.split("/")
            tool_instance = SimulatedTool(name, tools_info)
            self.tool_list[name] = tool_instance
            self.tools.append(tool_instance.get_tool_call_format())
            self.num_tools = len(self.tools)

    def pre_select_tools(self, tool_names):
        pre_selected_tools = []
        for tool_name in tool_names:
            for tool in self.tools:
                if tool["function"]["name"] == tool_name:
                    pre_selected_tools.append(tool)
                    break
        return pre_selected_tools

    def setup_logger(self):
        logger = AgentLogger(self.agent_name, self.log_mode)
        return logger

    def load_config(self):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        config_file = os.path.join(script_dir, self.agent_name, "config.json")
        with open(config_file, "r") as f:
            config = json.load(f)
            return config

    def get_response(self,
            query,
            temperature=0.0
        ):
        thread = CustomizedThread(target=self.query_loop, args=(query, ))
        thread.start()
        return thread.join()

    def query_loop(self, query):
        agent_process = self.create_agent_request(query)
        completed_response, start_times, end_times, waiting_times, turnaround_times = "", [], [], [], []
        while agent_process.get_status() != "done":
            thread = Thread(target=self.listen, args=(agent_process, ))
            current_time = time.time()
            # reinitialize agent status
            agent_process.set_created_time(current_time)
            agent_process.set_response(None)
            LLMRequestQueue.add_message(agent_process)
            thread.start()
            thread.join()
            completed_response = agent_process.get_response()
            if agent_process.get_status() != "done":
                self.logger.log(
                    f"Suspended due to the reach of time limit ({agent_process.get_time_limit()}s). Current result is: {completed_response.response_message}\n",
                    level="suspending"
                )
            start_time = agent_process.get_start_time()
            end_time = agent_process.get_end_time()
            waiting_time = start_time - agent_process.get_created_time()
            turnaround_time = end_time - agent_process.get_created_time()

            start_times.append(start_time)
            end_times.append(end_time)
            waiting_times.append(waiting_time)
            turnaround_times.append(turnaround_time)
 
        return completed_response, start_times, end_times, waiting_times, turnaround_times

    def create_agent_request(self, query):
        agent_process = self.agent_process_factory.activate_agent_process(
            agent_name = self.agent_name,
            query = query
        )
        agent_process.set_created_time(time.time())
        # print("Already put into the queue")
        return agent_process
    
    def listen(self, agent_process: AgentProcess):
        """Response Listener for agent

        Args:
            agent_process (AgentProcess): Listened AgentProcess

        Returns:
            str: LLM response of Agent Process
        """
        while agent_process.get_response() is None:
            time.sleep(0.2)

        return agent_process.get_response()

    def set_aid(self, aid):
        self.aid = aid

    def get_aid(self):
        return self.aid

    def get_agent_name(self):
        return self.agent_name

    def set_status(self, status):

        """
        Status type: Waiting, Running, Done, Inactive
        """
        self.status = status

    def get_status(self):
        return self.status

    def set_created_time(self, time):
        self.created_time = time

    def get_created_time(self):
        return self.created_time

    def set_start_time(self, time):
        self.start_time = time

    def get_start_time(self):
        return self.start_time

    def set_end_time(self, time):
        self.end_time = time

    def get_end_time(self):
        return self.end_time
