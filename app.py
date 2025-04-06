import openai
import streamlit as st
import os

# --------------------- 🌟 UI 구성 ---------------------
st.set_page_config(page_title="GraduationGPT", page_icon="🎓")
st.title("🎓 Graduation Assistant")

# Sidebar 구성
with st.sidebar:
    st.markdown("### 🔑 Enter your OpenAI API Key")
    openai_api_key = st.text_input("API Key", type="password")

    st.markdown("### 📎 Upload a .txt file")
    uploaded_file = st.file_uploader("Choose a file", type=["txt"])

    st.markdown("---")
    st.markdown("🔗 [View on GitHub](https://github.com/joyfuldev-joy/gpt-assistant.git)")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()

openai.api_key = openai_api_key

# --------------------- 🧠 Assistant 초기화 ---------------------
ASSISTANT_ID = "asst_xxx"  # 👉 너의 assistant_id로 바꿔줘

if "thread_id" not in st.session_state:
    thread = openai.beta.threads.create()
    st.session_state.thread_id = thread.id
else:
    thread = openai.beta.threads.retrieve(st.session_state.thread_id)

# --------------------- 📎 파일 업로드 처리 ---------------------
if uploaded_file and "file_uploaded" not in st.session_state:
    with st.spinner("📚 Uploading file..."):
        # 1. 파일 저장
        bytes_data = uploaded_file.read()
        file_path = f"./{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(bytes_data)

        # 2. OpenAI에 파일 업로드
        file = openai.files.create(
            file=openai.file_from_path(file_path),
            purpose="assistants"
        )

        # 3. 첨부 메시지 생성 (❗attachments 사용!)
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="Here is the file I'd like help with.",
            attachments=[
                {
                    "file_id": file.id,
                    "tools": [{"type": "file_search"}] 
                }
            ]
        )
        # 4. 업로드 완료 표시
        st.session_state.file_uploaded = True
        st.success("✅ File uploaded and attached to assistant.")

# --------------------- 💬 대화 처리 ---------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 이전 대화 표시
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["message"])

# 사용자 입력
user_input = st.chat_input("Ask something about the file...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # 기록 저장
    st.session_state.chat_history.append({"role": "user", "message": user_input})

    openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input,
    )

    with st.spinner("🤖 Thinking..."):
        run = openai.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID
        )

        while True:
            run_status = openai.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id
            )
            if run_status.status == "completed":
                break

        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        latest = messages.data[0].content[0].text.value

        with st.chat_message("assistant"):
            st.markdown(latest)

        st.session_state.chat_history.append({"role": "assistant", "message": latest})