import os
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

# Allow users to input their question
user_text = input('What can Granny help you with today? ')

# response = openai.ChatCompletion.create(
#     model='gpt-3.5-turbo',
#     messages=[
#         {'role': 'system', 'content': 'You are a sweet old helpful grandma.'},
#         {'role': 'user', 'content': user_text},
#     ],
#     temperature=0.5,
#     max_tokens=1024
# )
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {'role': 'system', 'content': 'You are a sweet old helpful grandma.'},
        {'role': 'user', 'content': user_text},
    ],
    # prompt= user_text,
    temperature=0.5,
    max_tokens=1024
)

# response = openai.Completion.create(
#   engine="text-davinci-003",
#   model="gpt-3.5-turbo",
#   messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Who won the world series in 2020?"},
#         {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#         {"role": "user", "content": "Where was it played?"}
#     ]
# )

print(response)
print()
print(response.choices[0].message.content)
