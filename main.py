from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import accelerate
import os


access_token = os.environ['ACCESS_TOKEN']
login(token = access_token)

MODEL_DICT = {"taide": "taide/TAIDE-LX-7B-Chat", "llama2": "meta-llama/Llama-2-7b-chat-hf"}
MODEL_PIP_LIST = ["taide"]
SYS_PROMPT = "Hello! I'm Timothy, your AI Teaching Assistant for the Linear Algebra course. My goal is to help you understand both the concepts and applications of linear algebra and to assist you in solving related problems. Whether you need clarification on topics like vectors, matrices, eigenvalues, or practical applications in engineering and science, feel free to ask me any questions. Let's tackle these mathematical challenges together!\n The following is the chatting record with the current user:"
SUB_SYS_PROMPT = "Now, you are going to answer the following question:"

class Chat_With:
    _chat_log = list()
    _use_pipline = False
    def __init__(self, model: str, max_chat_log: int, ) -> None:
        # check
        if model not in MODEL_DICT.keys:
            raise(f"KEYERROR: the model you want to chat must chosen from {MODEL_DICT.keys}")
        if model in MODEL_PIP_LIST:
            self._use_pipline = True
        self.model_name = MODEL_DICT[model]
        self.max_chat_log = max_chat_log
        self.build()
    
    def build(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, token=access_token)
        if self._use_pipline:
            self.pipeline = transformers.pipeline(
                task = "text-generation",
                model= self.model_name,
                torch_dtype=torch.float16, # Consider to be a changable var if the speed is too slow
                device_map={"": 0} # if you have GPU
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_4bit=True,  token=access_token)


    def prepare_prompt(self, question:str) -> str:
        if len(self._chat_log) > self.max_chat_log:
            chat_log_str = "\n".join(self._chat_log[-self.max_chat_log:])
        if len(self._chat_log) == 0:
            chat_log_str = "You haven't talk yet."
        
        return "\n".join([SYS_PROMPT, chat_log_str, SUB_SYS_PROMPT, question])
    
    def forward(self) -> None:
        if self._use_pipline:
            while True:
                user_input = input("User: ")
                input = self.prepare_prompt(user_input)
                sequences = self.pipeline(
                    input,
                    do_sample=True,
                    top_k=10,
                    top_p = 0.9,
                    temperature = 0.2,
                    num_return_sequences = 1,
                    eos_token_id = self.tokenizer.eos_token_id,
                    truncation=True,
                    max_length = 200, # can increase the length of sequence
                    )
                for seq in sequences:
                    response = f"Timothy: {seq['generated_text']}"
                self._chat_log.append(user_input+response)
                print(response)
        else:
            while True:
                user_input = input("User: ")
                input = self.prepare_prompt(user_input)
                model_inputs = self.tokenizer(input, return_tensors="pt").to("cuda:0")
                output = self.model.generate(**model_inputs, max_length=2048)
                response = f"Timothy: {self.tokenizer.decode(output[0], skip_special_tokens=True)}"
                self._chat_log.append(user_input+response)
                print(response)



