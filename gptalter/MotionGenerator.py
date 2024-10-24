import threading
import pyalter
import openai
import nest_asyncio
import asyncio
import logging
import tiktoken
import os ,sys 
import numpy as np
import ast, re, time
import shutil
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor

class MotionGenerator:
    def __init__(self, API, gpt3 = "gpt-4o-mini", gpt4 = "gpt-4-turbo", save = False, show=False, serial_device = "/dev/tty.usbserial-FT2KYV79"):
        super(MotionGenerator, self).__init__()
        nest_asyncio.apply()
        openai.api_key = API
        self.gpt4 = ChatOpenAI(temperature=0.7,model_name=gpt4, openai_api_key = openai.api_key)
        self.gpt4_low_temp = ChatOpenAI(temperature=0.3, model_name=gpt4, openai_api_key = openai.api_key)
        self.gpt3 = ChatOpenAI(temperature=0.2,model_name=gpt3, openai_api_key = openai.api_key)
        os.environ["OPENAI_API_KEY"] = openai.api_key
        self.show = show
        self.save = save
        self.prompt_1= """
        Your task is to describe exaggerate emotional expressions and facial expressions that accompany the content of the conversation. Output motion description should be several simple motions that the android is capable of. In addition, please create a facial expression that matches the input at the beginning. The android can only move its upper body and has the same joints as a human. Output shoul be written as much detail as possible.\

        Exammple1:\
        '''
        input : ["pretend the ghost"]
        discription :
        [
            "0 Create a wide-eyed facial expression of fear, mouth opened in a silent scream",
            "1 Quickly lean backward, as if startled by a sudden apparition",
            "2 Raise both hands and flutter them around the face, mimicking a ghostly movement",
            "3 Open mouth wide and shake head, showing a dramatic reaction of fear",
            "4 Move upper body from side to side, as if being unsettled by the ghostly presence",
            "5 Clench hands in front of chest, demonstrating extreme anxiety",
            "6 Dart eyes from side to side, as if witnessing an eerie activity",
            "7 Lean forward and then backward, copying the floating movement of a ghost",
            "8 Slowly return to rest position while maintaining a terrified expression"
        ]
        '''

        Exammple2:\
        '''
        input : ["I was enjoying a movie while eating popcorn at the theater when I realized that I was actually eating the popcorn of the person next to me."]
        discription :
        [
            "0 Create a shocked and entertained facial expression, eyes wide and mouth slightly open",  
            "1 Lean forward as if shocked and amused by the story",
            "2 Mimic the action of holding and eating popcorn with wide, exaggerated movements",  
            "3 Pause midway, freeze in place with a hand 'holding popcorn' in mid-air",  
            "4 Turn head sharply to the side, as if just realizing the mistake",  
            "5 Quickly pull hand back towards body in a dramatic recoil",  
            "6 Cover mouth with other hand, showing embarrassment and surprise",  
            "7 Shake head vigorously, as if in disbelief of the action",  
            "8 Lean back, laughing loudly and slapping knee in exaggerated amusement",  
            "9 Slowly wipe away 'tears' of laughter and return to rest position with a wide, amused grin"
        ]
        '''

        Guidelines:\
        '''
        1: Output should be list. Dont write "discription :"\
        2: Write as much detail as possible. Describe action step by step. Do not write any explanation.Describe exaggeration as much as possible.\
        3: Android has only upper body.\
        4: Create a facial expression that matches the input at the beginning
        '''

        input :{input}
        """

        self.prompt_2 = """
        Write python code to operate an android named Alter3. Here's what you need to know.\

        Alter3 has 42 joints throughout its body, numbered from 1 to 42. You can move a joint by specifying its number and sending a signal. For instance, to move joints number 1,2,3, use: ''' alter.set_axes([1,2,3], [255, 100, 127]) '''. The first argument is the joint number, and the second argument is a value between 0 and 255, specifying the joint angle. Each operation takes approximately 0.1~0.2 second, so insert '''time.sleep(0.2)''' between operations.\

        '''
        Alter3's Joints:
        - Axis 1: Eyebrows. 255 = down, 0 = up, 64 = neutral.
        - Axis 2: Pupils (horizontal). 255 = left, 0 = right, 140 = neutral.
        - Axis 3: Pupils (vertical) . 255 = up, 0 = down, 128 = neutral.
        - Axis 4: Eyes. 255 = closed, 0 = open.
        - Axis 5-6: Left/Right cheek. 255 = raised (smile), 0 = lowered.
        - Axis 7: Lips. 255 = puckered, 0 = relaxed.
        - Axis 8: Mouth. 255 = open, 0 = closed.
        - Axis 9: Head tilt. 255 = left, 0 = right, 128 = neutral.
        - Axis 10: Head up/down. 255 = down, 0 = up, 160 = neutral.
        - Axis 11: Head rotate. 255 = left, 0 = right, 122 = neutral.
        - Axis 12: Neck nod. 255 = backward, 0 = forward, 128 = neutral.
        - Axis 13: Hips tilt. 255 = left, 0 = right, 128 = neutral.
        - Axis 14: Waist bend. 255 = backward, 0 = forward, 128 = neutral.
        - Axis 15: Abdomen rotation. 255 = left, 0 = right, 128 = neutral.
        - Axis 16: Left shoulder up/down. 255 = up, 0 = down, 128 = neutral.
        - Axis 17: Left shoulder forward/back. 255 = forward, 0 = back, 64 = neutral.
        - Axis 18: Left armpit open/close. 255 = open, 0 = close, 64 = neutral.
        - Axis 19: Left arm lift. 255 = up, 0 = down, 64 = neutral.
        - Axis 20: Left upper arm rotation. 255 = left, 0 = right, 32 = neutral.
        - Axis 21: Left elbow bend. 255 = bent, 0 = straight, 32 = neutral.
        - Axis 22: Left forearm twist. 255 = downside, 0 = upside, 128 = neutral.
        - Axis 23: Left wrist bend. 255 = straight, 0 = bent, 128 = neutral.
        - Axis 24: Left wrist side bend. 255 = left, 0 = right.
        - Axis 25: Left thumb. 255 = bent, 0 = spread.
        - Axis 26: Left index finger. 255 = bent, 0 = spread.
        - Axis 27: Left middle finger. 255 = bent, 0 = spread.
        - Axis 28: Left ring/little fingers. 255 = bent, 0 = spread.
        - Axis 29: Right shoulder up/down. 255 = up, 0 = down, 128 = neutral.
        - Axis 30: Right shoulder forward/back. 255 = forward, 0 = back, 64 = neutral.
        - Axis 31: Right armpit open/close. 255 = open, 0 = close, 64 = neutral.
        - Axis 32: Right arm lift. 255 = up, 0 = down, 64 = neutral.
        - Axis 33: Right upper arm rotation. 255 = right, 0 = left, 32 = neutral.
        - Axis 34: Right elbow bend. 255 = bent, 0 = straight, 32 = neutral.
        - Axis 35: Right forearm twist. 255 = downside, 0 = upside, 128 = neutral.
        - Axis 36: Right wrist bend. 255 = straight, 0 = bent, 128 = neutral.
        - Axis 37: Right wrist side bend. 255 = right, 0 = left.
        - Axis 38: Right thumb. 255 = bent, 0 = spread.
        - Axis 39: Right index finger. 255 = bent, 0 = spread.
        - Axis 40: Right middle finger. 255 = bent, 0 = spread.
        - Axis 41: Right ring/little fingers. 255 = bent, 0 = spread.
        - Axis 42: Whole body raise/lower. 255 = raised, 0 = lowered, 128 = neutral.
        '''

        OUTPUT Example1: "0 Lean forward aggressively, as if ready to dive into the music",\
        '''
        ### 0

        # Eyebrows to furrowed
        alter.set_axes([1], [255])
        time.sleep(0.1)

        # Set pupils to neutral position
        alter.set_axes([2, 3], [140, 128])
        time.sleep(0.1)

        # Open eyes wide
        alter.set_axes([4], [0])
        time.sleep(0.1)

        # Tilt the hips forward
        alter.set_axes([13], [0])
        time.sleep(0.1)

        # Do some repeat action
        for i in range(3):
            alter.set_axes([13], [0])
            time.sleep(0.5)
            alter.set_axes([13], [255])
        '''

        Guidelines:\
        1: Output should be only python code. Do not insert any syntax highlighting like ```.
        2: The input begins with a number, such as "0" or "2". The output should start with "### 0" or "### 2" depending on the number.
        3: Do not Create an instance of alter3.
        4: Do not insert python syntax highlighting like ```python ```.
        5: Do not write "import alter".
        6: Use # and write short description of code.

        input: {input}
        """
        self.alter = pyalter.Alter3("serial", serial_port=serial_device)
        self.all_axes  = list(np.arange(1,44))
        self.initial_value = [64,140,128,0,0,0,0,0,128,160,122,128,128,128,128,128,64,64,64,32,32,128,128,0,0,0,0,0,64,64,64,64,32,32,128,128,0,0,0,0,0,128,185]
        self.new_directory_path = None
        self.recipe = None
        self.motion_dir = self.create_dir()

    def create_dir(self):
        current_directory = os.getcwd()
        motion_dir = os.path.join(current_directory, "motions")

        if not os.path.exists(motion_dir):
            os.makedirs(motion_dir)
            print(f"'{motion_dir}' が作成されました✨")
        else:
            print(f"'{motion_dir}' はすでに存在します😊")
        return motion_dir

    def prompt1(self, message):
        PROMPT_1 =  PromptTemplate.from_template(self.prompt_1)
        recipe_gen = PROMPT_1 | self.gpt3
        response = recipe_gen.invoke({"input": message}).content
        if self.show == True:
            print(f"\033[94m generating recipe \033[0m")
        try:
            self.recipe = ast.literal_eval(response)
        except Exception as e:
            print(f"Failed to parse recipe: {e}")
            return None, None
        safe_message = (message[:25]) if len(message) > 25 else message
        self.new_directory_path = f"{self.motion_dir}/{safe_message}"
        if not os.path.exists(self.new_directory_path):
            os.mkdir(self.new_directory_path)
        if self.show == True:
            for i in range(len(self.recipe)):
                print(f"\033[94m{self.recipe[i]}\033[0m")
        return self.recipe, self.new_directory_path
    
    def get_gpt_response(self,systemprompt: str, new_directory_path) -> str:
        PROMPT_2 =  PromptTemplate.from_template(self.prompt_2)
        code_gen = PROMPT_2 | self.gpt4_low_temp
        code = code_gen.invoke({"input": systemprompt}).content
        num = re.findall(r'\d+', code[0:10])
        if self.show == True:
            print(f"\033[94m {num} -> DONE \033[0m")
        with open(f"{new_directory_path}/{num[0]}.txt", 'w', encoding='utf-8') as f:
            f.write(code)
        return code

    def async_gpt_responses(self, recipe, new_directory_path):
        loop = asyncio.get_event_loop()
        responses = []
        with ThreadPoolExecutor() as executor:
            tasks = [loop.run_in_executor(executor, self.get_gpt_response, line, new_directory_path) for line in recipe]
            responses = loop.run_until_complete(asyncio.gather(*tasks))
        return responses

    def prompt2(self):
        start_time = time.time()
        if self.show == True:
            print(f"\033[92m TIME: {time.time()-start_time} \033[0m ")
        responses = self.async_gpt_responses(self.recipe, self.new_directory_path)
        if self.show == True:
            print(f" \033[92m TIME: {time.time()-start_time} \033[0m ")

    def identify_code_blocks_by_newlines(self, content):
        blocks = []
        code_block = []
        in_code_block = False 

        for line in content:
            stripped_line = line.strip()
            if stripped_line:
                if stripped_line.startswith("```"):
                    line = "#"+ line
                if stripped_line.startswith("while"):
                    line = "#"+ line
                code_block.append(line)
                in_code_block = True
            elif in_code_block:
                if line.startswith("```"):
                    line = "#"+ line
                if line.startswith("while"):
                    line = "#"+ line
                code_block.append(line)
            else:  # end of a code block
                if code_block:
                    blocks.append("".join(code_block))
                    code_block = []
                    in_code_block = False
        if code_block:
            blocks.append("".join(code_block))
        return blocks

    def execute_code(self, path = None):
        if path == None:
            path = self.new_directory_path
        else:
            path = path
        self.alter.set_axes(self.all_axes, self.initial_value)
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        files.sort(key=lambda f: int(f.split('.')[0]))
        for filename in files:
            with open(os.path.join(self.new_directory_path, filename), 'r') as file:
                content = file.read()
                code_blocks = self.identify_code_blocks_by_newlines(content)
                for index, code in enumerate(code_blocks):
                    try:
                        exec_locals = {}
                        exec(code, globals(), exec_locals)
                    except Exception as e:
                        print(code)
                        print(f"Error in block {index + 1}: {str(e)}")
        with open(f"{path}/recipe.txt", 'a', encoding='utf-8') as f:
            for item in self.recipe:
                f.write(f"{item}\n")
        self.alter.set_axes(self.all_axes, self.initial_value)
        if self.save == False:
            shutil.rmtree(path)

    def run(self, message):
        self.recipe, self.news_directory_path = self.prompt1(message)
        self.prompt2()
        self.execute_code()
