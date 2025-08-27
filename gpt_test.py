# import os
# import openai

# # APIキーを環境変数から取得
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # ChatGPT-4oに質問
# response = openai.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "PythonでHello Worldを表示するコードを書いてください"}
#     ],
#     temperature=0.7,  # 創造性の調整
#     max_tokens=200    # 出力トークン数の上限
# )

# # レスポンスの表示
# print(response.choices[0].message.content)



# from openai import OpenAI

# client = OpenAI(
#     base_url="https://api.aimlapi.com/v1",
#     api_key="fdbb4b28b48f42b7aea3bea1489bc085",
# )

# response = client.chat.completions.create(
#     model="openai/gpt-4o-",
#     messages=[
#         {
#   "role": "user",
#   "content": "日本語でお願いします"
# }
#     ],
#     temperature=0.7,
#     top_p=0.7,
#     frequency_penalty=1,
#     max_tokens=512,
#     # top_k=50,
# )

# message = response.choices[0].message.content
# print(f"Assistant: {message}")


from openai import OpenAI

client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    # api_key="fdbb4b28b48f42bs89bc085",
    api_key="fdbb4b28b48f42b7aea3bea1489bc085",
    
)

response = client.chat.completions.create(
    model="google/gemma-3-12b-it",
    messages=[
        {
  "role": "user",
  "content": "日本語で答えて"
}
    ],
    temperature=0.7,
    top_p=0.7,
    frequency_penalty=1,
    max_tokens=512,
    # top_k=50,
)

message = response.choices[0].message.content
print(f"Assistant: {message}")

    