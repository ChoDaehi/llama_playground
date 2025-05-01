from transformers import AutoProcessor, AutoModelForImageTextToText,BitsAndBytesConfig,AutoConfig
import torch
from accelerate import init_empty_weights, infer_auto_device_map

# 모델 불러오기
# 4ビット量子化の設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model_name = "meta-llama/Llama-4-Scout-17B-16E"
config = AutoConfig.from_pretrained(model_name)

with init_empty_weights():
    model = AutoModelForImageTextToText.from_config(config)

# GPU 0에 최대 20GiB, GPU 1에도 분산
device_map = infer_auto_device_map(
    model,
    max_memory={
        "cuda:0": "22GiB",
        "cuda:1": "22GiB",
        "cuda:2": "22GiB",
        "cuda:3": "22GiB" # 다른 GPU가 있다면 추가
    },
    no_split_module_classes=["LlamaDecoderLayer"]  # 나누기 싫은 레이어 지정
)

# 실제 모델 로드
processor = AutoProcessor.from_pretrained(model_name,device_map=device_map)
model = AutoModelForImageTextToText.from_pretrained(model_name, quantization_config=bnb_config,device_map=device_map)


# 채팅 함수
def chat_with_model(user_input: str):
    """
    모델과의 채팅 인터페이스를 구현하는 함수.

    Args:
        user_input (str): 사용자가 입력한 질문 또는 메시지.

    Returns:
        str: 모델이 생성한 응답.
    """
    try:
        # 사용자 입력을 처리
        inputs = processor(text=user_input, return_tensors="pt").to("cuda")

        # 모델을 통해 응답 생성
        outputs = model.generate(**inputs)

        # 응답 디코딩
        decoded_output = processor.decode(outputs[0], skip_special_tokens=True)

        return decoded_output
    except Exception as e:
        return f"오류가 발생했습니다: {e}"


# 채팅 예시
if __name__ == "__main__":
    print("안녕하세요! 모델과 대화를 시작하세요. 'exit'를 입력하면 종료됩니다.")

    while True:
        user_message = input("나: ")

        if user_message.lower() == "exit":
            print("채팅을 종료합니다. 감사합니다!")
            break

        # 모델로부터 응답받기
        response = chat_with_model(user_message)
        print(f"모델: {response}")
