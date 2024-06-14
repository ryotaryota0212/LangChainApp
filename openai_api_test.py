import os

import openai

# 環境変数からAPIキーを取得
api_key = os.getenv("OPENAI_API_KEY")

# OpenAIクライアントを初期化
openai.api_key = api_key

# 複数の temperature 値を設定
temperature_values = [1.1, 1.2]

# 各 temperature 値でテキストを生成して表示
for temperature in temperature_values:
    response = openai.chat.completions.create(
      model="gpt-4o",
      messages=[
        {
          "role": "user",
          "content": "むかしむかし、あるところに"
        }
      ],
      temperature=temperature,
      max_tokens=2048,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    # 結果の表示
    print(f"Temperature: {temperature}")
    print(f"トータルトークン数: {response.usage.total_tokens}")
    print(response.choices[0].message.content)
    print("\n" + "="*80 + "\n")
