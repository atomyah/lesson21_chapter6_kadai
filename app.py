## ライブラリのインポート: pip install langchain==0.3.0 openai==1.47.0 langchain-community==0.3.0 langchain-openai==0.2.2 httpx==0.27.2
import os
from dotenv import load_dotenv  ## pip install python-dotenvインストール必要
import streamlit as st
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

st.title("サンプルアプリ: 二人の専門家に相談しよう！")
st.write("##### 専門家1: 医療と先端治療の専門家")
st.write("医療界の先端技術と最新情報を聞くことができます。")
st.write("##### 専門家2: スピリチュアルの専門家")
st.write("スピリチュアルと宇宙の仕組みを聞くことができます。")


selected_item = st.radio(
    "専門家を選択してください。",
    ["医療の専門家", "スピリチュアルの専門家"]
)

st.divider()    ## 区切り線

## チャットモデルの初期化
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

# メモリの初期化をセッション状態で管理
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,   ## # メモリの初期化時に返信の形式を指定
        memory_key="chat_history",
        max_token_limit=1000
    )

system_template = "あなたは{genre}の専門家です。質問に対して初心者にも分かりやすく簡潔に答えてください。{genre}以外の質問には答えないでください。"
human_template = "ユーザー：{question}"

## ChatPromptTemplateを生成
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    MessagesPlaceholder(  # 履歴用のプレースホルダーを追加
        variable_name="chat_history"
    ),
    HumanMessagePromptTemplate.from_template(human_template),
])


# 以降のコードでst.session_state.memoryを使用してメモリを操作
if selected_item == "医療の専門家":
    st.write("医療の専門家に質問してみましょう。")

    question = st.chat_input("質問を入力してください。")
    if question:
        with st.spinner("考え中..."):
            # メモリから履歴を取得
            chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]

            messages = prompt.format_prompt(
                genre="医療", 
                question=question,
                chat_history=chat_history
            ).to_messages()
            result = llm.invoke(messages)

            # 会話を保存（人間の質問とAIの回答）
            st.session_state.memory.save_context(
                {"input": question},
                {"output": result.content}
            )

            st.write("あなたの質問: {question}".format(question=question))
            st.write(f"医療の専門家の回答: {result.content}")

elif selected_item == "スピリチュアルの専門家":
    st.write("スピリチュアルの専門家に質問してみましょう。")

    question = st.chat_input("質問を入力してください。")
    if question:
        with st.spinner("考え中..."):
            # メモリから履歴を取得
            chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]

            messages = prompt.format_prompt(
                genre="スピリチュアル", 
                question=question,
                chat_history=chat_history
            ).to_messages()
            result = llm.invoke(messages)

            # 会話を保存（人間の質問とAIの回答）
            st.session_state.memory.save_context(
                {"input": question},
                {"output": result.content}
            )

            st.write("あなたの質問: {question}".format(question=question))
            st.write(f"スピリチュアルの専門家の回答: {result.content}")

# 会話履歴を表示
print("\n=== 会話履歴 ===")
conversation_history = st.session_state.memory.load_memory_variables({})["chat_history"]
print(conversation_history)