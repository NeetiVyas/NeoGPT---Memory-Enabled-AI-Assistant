# Main Streamlit application
import mimetypes
import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import io
import os
import requests
import base64
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from duckduckgo_search import DDGS
from langchain.tools import Tool
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from mem0 import Memory
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

load_dotenv()

API_BASE = "http://127.0.0.1:8000"

qdrant_url=os.getenv("QDRANT_URL")
qdrant_api_key=os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=qdrant_url,  
    api_key=qdrant_api_key,
)

uri=os.getenv("NEO4J_URL")
username=os.getenv("NEO4J_USERNAME")
password=os.getenv("NEO4J_PASSWORD")

with GraphDatabase.driver(uri, auth=(username, password)) as driver:
    driver.verify_connectivity()

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", transport="rest" )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 200
)

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "url": qdrant_url,
            "api_key": qdrant_api_key,
            "collection_name": "mem0_default",
        }
    },
    "graph": {
        "provider": "neo4j",
        "config": {
            "url": uri,
            "username": username,
            "password": password
        }
    },
    "llm": {
        "provider": "together",
        "config": {
            "model": "mistralai/Mistral-7B-Instruct-v0.2" 
        }
    },
    "embedder": {
        "provider": "gemini", 
        "config": {
            "model": "models/gemini-embedding-001",
        }
    }
}

mem_client = Memory.from_config(config)

st.set_page_config(
    page_title="SnackGPT - Your Recipe Assistant",
    page_icon="🍽️",
)

@tool
def ddg_search(query: str, max_results: int = 4) -> str:
    """
    Use this tool when you need to search something from Web.
    Input: a natural language query. Output: summarized top results.
    Fetch and format Duckduckgo search results.
    """

    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as e:
        return f"Search Failed {e}"
    
    if not results:
        return "No results found"

    out_lines = []
    for i,r in enumerate(results[:max_results], start=1):
        title = r.get("title") or ""
        snippet = r.get("body") or r.get("snippet") or ""
        href = r.get("href") or r.get("url") or ""

        if len(snippet) > 300:
            snippet = snippet[:300].rstrip() + '...'
            out_lines.append(f"{i}. {title}\n{snippet}\n{href}")
            return '\n\n'.join(out_lines)
        
@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f'https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}'

    response = requests.get(url)

    return response.json()


# CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Caprasimo&family=Oswald:wght@200..700&family=Playwrite+AU+QLD:wght@100..400&display=swap');
        @font-face {
            font-family: Caprasimo;
            src : url('Caprasimo/Caprasimo-Regular.ttf') format('truetype');
        }
        [data-testid="stHeader"] {
            min-height: 2.2em !important;
            height: 2.2em !important;
            padding-top: 0.2em !important;
            padding-bottom: 0.2em !important;
        }
        [data-testid="stToolbar"] {
            min-height: 1.8em !important;
            height: 1.8em !important;
            padding: 0.2em 0.5em !important;
        }
        /* Move app content to top */
        [data-testid="stAppViewContainer"] {
            margin-top: 0;
            padding-top: 0;
        }
        /*
        [data-testid="stLayoutWrapper"] [data-testid="stChatMessage"] {
            background-color: transparent !important;
            box-shadow: none !important;
        }
        [data-testid="stLayoutWrapper"] [data-testid="stChatMessage"] p {
            padding: 0.3rem 0.5rem !important;
        }
        [data-testid="stBottom"] [data-testid="stBottomBlockContainer"] {
            background-color: lightblue;
        }
        [data-testid="stDecoration"] {
            background-color: lightblue !important;
        }
        [data-testid="stBottom"] {
            background-color: lightblue !important;
        }*/
        [data-testid="stLayoutWrapper"][data-testid="stChatMessage-user"] {
            max-width: 90% !important; 
            margin-left: auto; 
            margin-right: 0;   
        }
            .header {
                text-align: center;
                margin-top: -9vw;
                color: #3A508C;
                font-family: 'Caprasimo', cursive !important;
                letter-spacing: 2px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                font-size: 5vw;
                font-weight: bold;
                padding: 6px;
            }
            .header img {
                width: 16%;
                margin-top: -4vw;
            }
            .assistant-message {
                background-color: #e6f0fb;
                color: black;
                text-align: left;
                margin: 10px 4px;
                width: fit-content;
                max-width: 90%;
                font-size: 14px;
                word-wrap: break-word;
                padding: 8px;
                border-radius: 0 8px 8px 8px;
            }
            .user-message {
                background-color: #e8f5e9;
                padding: 8px;
                border-radius: 0 8px 8px 8px;
                margin: 10px 0;
                color: black;
                text-align: right;
                width: fit-content;
                max-width: 80%;
                font-size: 14px;
                margin-left: auto;
                word-wrap: break-word;
            }
            div[data-testid="stButton"] {
                display: inline-block !important;
                vertical-align: middle !important;
                width: auto !important;
                margin: 0 4px 0 0 !important;
                padding: 0 !important;
            }

            .history-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 6px;
                margin-bottom: 4px;
            }

            .history-label button {
                font-size: 12px !important;
                padding: 2px 6px !important;
                border-radius: 6px !important;
                width: auto !important;
            }
    </style>
""", unsafe_allow_html=True)

if 'access' not in st.session_state:
    st.session_state.access = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = None


with st.sidebar.expander("🔍 Search Chats"):
    query = st.text_input("Enter Search Query")
    if query.strip():
        headers={"Authorization": f"Bearer {st.session_state['access']}"}
        resp = requests.get(f"{API_BASE}/search/", params={"q": query}, headers=headers)
        print("Search API \n", resp.text)
        if resp.status_code==200:
            results=resp.json()
            if results:
                for r in results:
                    if st.button(r["message"][:40], key=f"res_{r['id']}"):
                        st.session_state.session_id = r['session_id']
                        chat_resp = requests.get(f"{API_BASE}/chat/?session_id={r['session_id']}", headers=headers)
                        if chat_resp.status_code==200:
                            st.session_state.messages = [
                                {"role": m["role"], "content": m["message"]}
                                for m in chat_resp.json()
                            ]
                            st.rerun()
            else:
                st.write("NO results found")
        else:
            st.error("Search Failed")

st.sidebar.title("🔐 Account")
st.sidebar.subheader("Mode")
mode = st.sidebar.radio("", ["Guest", "Signup", "Login"])

if mode=='Signup':
    username = st.sidebar.text_input("Username")
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password")
    if st.sidebar.button("Signup"):
        st.session_state.clear()
        resp = requests.post(f"{API_BASE}/signup/", json={"username": username, "email": email, "password": password})
        print("Signup request: ", resp)
        if resp.status_code == 201:
            st.sidebar.success("✅ Signup successful! Please login.")
        else:
            st.sidebar.error(resp.text)

if mode=='Login':
    email = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        st.session_state.clear()
        resp = requests.post(f"{API_BASE}/login/", json={"username": email, "password": password})
        print("Post Login Token generation ", resp)
        print('\n', resp.json())
        if resp.status_code == 200:
            st.session_state["username"] = email
            st.session_state["password"] = password
            st.session_state["access"] = resp.json()["access"]
            st.session_state['refresh'] = resp.json()['refresh']
            st.sidebar.success("✅ Logged In")
            st.rerun()
        else:
            print(resp.json())
            print('\n', resp.text)
            st.sidebar.error("❌ Login failed")


#if st.session_state.access:  (if i use this even though access is set to None , it will return True and for unautheticated user Logout button will be available)
if st.session_state.get("access"):  #(will only return true when there is access token)
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.sidebar.success("Logged out successfully.")
        st.rerun()
    

# Fetch history if logged in
if "access" in st.session_state:
    headers = {"Authorization": f"Bearer {st.session_state['access']}"}
    #fetches all records of a user
    resp = requests.get(f"{API_BASE}/chat/", headers=headers)
    print(resp)
    if resp.status_code == 200:
        sessions = {}
        #grouping the fetched records based on session_id
        for msg in resp.json():
            sid = msg["session_id"]
            print("Session ", sid)
            if sid not in sessions:
                sessions[sid] = []
            sessions[sid].append(msg)
            
        st.sidebar.subheader("History")

        #creating different buttons for session
        for sid, messages in sessions.items():
            label = messages[0]['message'][:20] + '...' if messages else f"Session {sid[:5]}"
            #Clicking a Session Button
            with st.sidebar.container():
                st.markdown('<div class="history-row">', unsafe_allow_html=True)
                st.markdown('<div class="history-label">', unsafe_allow_html=True)
                if st.button(label, key=sid):
                    st.session_state.session_id = sid
                    st.session_state.messages = [
                        {"role": m["role"], "content": m["message"]} for m in messages
                    ]
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="delete-btn">', unsafe_allow_html=True)
                if st.button("❌", key=f"delete_{sid}"):
                    del_resp = requests.delete(f"{API_BASE}/chat/?session_id={sid}", headers=headers)
                    if del_resp.status_code == 200:
                        st.rerun()
                    else:
                        st.error(del_resp.json().get("error", "Failed to delete"))
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to NeoGPT! 🤖✨ I'm here to answer your queries, explore ideas and even search the internet to bring you the best information."}
    ]

st.markdown(f"""<div class='header'>NeoGPT <img src='https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3OWtwNGx3cHNtOTZhYndlaDB1bHhuNnJtd2l3dWRhOTBlZWowNTlkeCZlcD12MV9zdGlja2Vyc19zZWFyY2gmY3Q9cw/Mn0PsxMyaoXRu/giphy.gif'></div>""", unsafe_allow_html=True)

if "llm" not in st.session_state:
    st.session_state.llm = ChatTogether(model="meta-llama/Llama-3-70b-chat-hf")

available_tools = [ddg_search, get_weather_data]
    
if "llm_with_tools" not in st.session_state:
    st.session_state.llm_with_tools = st.session_state.llm.bind_tools(available_tools)

llm_with_tools = st.session_state.llm_with_tools

import asyncio
import threading

# Create and set event loop if none exists
try:
    asyncio.get_running_loop()
except RuntimeError:  # No event loop in current thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

if "faiss_store" not in st.session_state:
    st.session_state.faiss_store = None

# Helper: Extract text from PDF
def extract_pdf_text(file):
    pdf_reader = PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text.strip()

# Helper: Convert image to base64 (for Gemini image input)
def get_image_content(file):
    image = Image.open(io.BytesIO(file.getvalue()))
    return image

def get_image_media(file):
    """Return Gemini-compatible media dict for any image file."""
    mime_type, _ = mimetypes.guess_type(file.name)
    if mime_type is None:
        mime_type = "image/png"  # fallback
    base64_data = base64.b64encode(file.getvalue()).decode("utf-8")
    return {"type": "media", "mime_type": mime_type, "data": base64_data}
 
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        if(msg["role"]=='assistant'):
            st.markdown(f"<div class='assistant-message'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)

combined_text = ""
agent_response = ""
    
if prompt := st.chat_input("Ask anything...", accept_file="multiple"):
    user_text = prompt.text.strip() if prompt.text else ""
    user_files = prompt.files if prompt.files else []

    combined_text = user_text
    st.session_state.image_inputs = []
    all_new_chunks = []

    # 1️⃣ Process files: extract text or prepare image for Gemini
    for file in user_files:
        file_name = file.name.lower()
        if file.type == "application/pdf" or file_name.endswith("pdf") :
            pdf_text = extract_pdf_text(file)
            chunks = text_splitter.split_text(pdf_text)
            all_new_chunks.extend(chunks)
            #combined_text += f"\n\n[Pdf Content Extracted]: \n{pdf_text}" if pdf_text else ""
        elif file.type.startswith("image/") or file_name.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff")):
            # Send images directly to Gemini
            st.session_state.image_inputs.append(get_image_media(file))
        else:
            combined_text += f"\n\n[Unsupported file type: {file.name}]"

    if user_files:
        cols = st.columns(min(len(user_files), 3))

    if all_new_chunks:
        st.spinner("Analysing the file...")
        if st.session_state.faiss_store is None:
            st.session_state.faiss_store = FAISS.from_texts(all_new_chunks, embedding_model)
        else:
            #add_texts is instance method & from_texts is class method.
            st.session_state.faiss_store.add_texts(all_new_chunks)
        all_new_chunks.clear()

    with st.chat_message("user"):
        display_content = combined_text if combined_text else f"Attached {len(user_files)} files"
        st.markdown(f"<div class='user-message'>{display_content}</div>", unsafe_allow_html=True)
        for idx ,file in enumerate(user_files):
            if file.type.startswith("image/") or file.name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff")):
                with cols[idx % 3]:
                    img_base64 = base64.b64encode(file.getvalue()).decode("utf-8")
                    img_html = f"<img src='data:image/png;base64,{img_base64}' class='chat-image'>"
                    st.markdown(img_html, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "user", "content": combined_text})
    st.rerun()

# If user asked a question (we manage it by reading the last message user added)
def last_user_query():
    for m in reversed(st.session_state.messages):
        if m["role"] == "user":
            return m['content']
    return None

last_query = last_user_query()

if last_query and st.session_state.messages[-1]['role'] == 'user':
    human_content=[]
    user_id = st.session_state.get("username", "guest")

    mem_result=mem_client.search(query=last_query, user_id=user_id)

    memories = '\n'.join([m['memory'] for m in mem_result.get("results", [])])

    if memories:
        print(f"Retrieved memories for user {user_id}:")
        memory_context = f"Important info about the user:\n{memories}\n\n"
        #human_content.append({"type": "text", "text": memory_context})
        print(memories)

    if last_query.strip():
        #human_content.append(memory_context)
        human_content.append({"type": "text", "text": last_query})
    if st.session_state.image_inputs:
        human_content.extend(st.session_state.image_inputs)
    
    history = [
                HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
                for m in st.session_state.messages if m["content"]
            ]

    with st.chat_message("assistant"):
        
        with st.spinner("generating response...."):
            try:
                agent_response = None
                memory_prompt=memory_prompt = f"""
                You are a memory-aware assistant with the ability to extract facts, reason with user context, 
                and decide when to use external tools.

                ### Your Capabilities:
                1. **Memory Awareness**  
                - You have access to the following user memories (may or may not be useful):  
                    {memory_context}  
                - Always consider these memories when responding to ensure personalized, consistent answers.  
                - If new important facts are shared, try to extract and summarize them in a structured way.  

                2. **Knowledge Handling**  
                - If the user query can be answered directly using your reasoning or stored memories, do so.  
                - If the query requires information that you cannot deduce from memory or general knowledge, 
                    use the available tools.  

                3. **Tool Usage**  
                You have access to these tools:  
                {available_tools}  

                - **DDG (DuckDuckGo)** → Use this when you need to search for factual, up-to-date, or external information.  
                - **get_weather_tool** → Use this when the user asks about current, forecast, or location-based weather.  

                ### Tool Usage Guidelines:
                - Do NOT use tools unnecessarily if you already know the answer.  
                - Always choose the most relevant tool based on the query.  
                - If the query is partially answerable with memory but requires fresh data, combine both.  

                4. **Response Formatting**  
                - If you use a tool, clearly integrate the tool’s result into your natural response.  
                - Never expose raw tool calls to the user, only the final useful information.  
                - Be concise, helpful, and user-focused.  

                ### Example Behavior:

                **Example 1:**  
                User: "Find the weather of the city Coimbatore"  
                Action: Call `get_weather_tool("Coimbatore")`  
                Response: "The current weather in Coimbatore is 28°C with light rain."

                **Example 2:**  
                User: "Who is the CEO of Tesla?"  
                Action: Call `DDG("CEO of Tesla")`  
                Response: "The current CEO of Tesla is Elon Musk."

                **Example 3:**  
                User: "You remember I told you I am learning Django, right?"  
                Action: Recall from memory → respond without tool.  
                Response: "Yes! You mentioned you are learning Django recently. Do you want me to share a roadmap or help with a specific issue?"
                ---

                Your goal is to seamlessly combine **memory, reasoning, and tool use** to provide accurate, helpful, and personalized responses.
                """

                if st.session_state.faiss_store:
                    retriever = st.session_state.faiss_store.as_retriever(search_kwargs={'k': 4})
                    retrieved_docs = retriever.get_relevant_documents(last_query)

                    if retrieved_docs:
                        context_text = '\n\n'.join([d.page_content for d in retrieved_docs])    
                        rag_prompt = [
                            SystemMessage(content='You are a helpful assistant. Use provided context if relevant.'),
                            HumanMessage(content=[{'type': 'text', 'text': f'Context \n{context_text}\n\n Question{last_query}'}] + st.session_state.image_inputs)
                        ]
                        agent_response = st.session_state.llm.invoke(rag_prompt).content
                    else:
                        agent_response=llm_with_tools.invoke([SystemMessage(content=memory_prompt)] + [SystemMessage(content="You are a helpful assistant that uses long-term memory when appropriate.")] + [HumanMessage(content=human_content)] + history).content
                else:
                    agent_response=llm_with_tools.invoke([SystemMessage(content=memory_prompt)] + [SystemMessage(content="You are a helpful assistant that uses long-term memory when appropriate.")] + [HumanMessage(content=human_content)] + history).content
            except Exception as e:
                agent_response = f"Error generating response: {e}"

            st.markdown(f"<div class='assistant-message'>{agent_response}</div>", unsafe_allow_html=True)

    # 3️⃣ Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": agent_response})
     
    messages=[
        {'role': 'user', 'content': combined_text},
        {"role": "assistant", "content": agent_response}
    ]

    mem_client.add(
        messages=messages,
        user_id=user_id,
    )

    # 4️⃣ Save to backend if logged in
    if "access" in st.session_state:
        post_data = {
            "message": last_query, 
            "assistant_message": agent_response
        }
        if st.session_state.get("session_id"):
            post_data["session_id"] = st.session_state["session_id"]

        print(post_data)

        chat_resp = requests.post(f"{API_BASE}/chat/", json=post_data, headers=headers)

        # On first POST, save session_id to state
        if chat_resp.status_code==201 and not st.session_state.get("session_id"):
            st.session_state["session_id"] = chat_resp.json().get("session_id")

    
    
