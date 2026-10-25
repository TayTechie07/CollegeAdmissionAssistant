
import os
from typing import TypedDict, Annotated
from langgraph .graph.message import add_messages
from langgraph.graph import StateGraph, START ,END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def build_retriever(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

academic_retriever = build_retriever("academics_handbook.pdf")
fee_retrievr = build_retriever("fee_structure.pdf")

##model
llm = ChatGroq(model = "openai/gpt-oss-120b", temperature=0.4)

##States
class State(TypedDict):
    programme : str
    messages : Annotated[list,add_messages]
    query_type : str
    retrieved_context : str

##Nodes

def classifier_node(state : State) -> dict:
    "Look at the latest user message and decide which path to take"
    last_messsage = state["messages"][-1].content


    prompt = f"""
        Classify the following student query into exactly one category:
'academic', 'fee', or 'general'.

Use these rules:
- academic: Use this category when the student asks about admissions,
  eligibility criteria, programs, courses, merit, exams, or grades.
- fee: Use this category when the student asks about tuition fees,
  scholarships, payment methods, dues, or refunds.
- general: Use this category for all other questions, such as
  university timings, contact information, campus facilities, or
  general university information.

Student query: {last_messsage}

Return only one category name: academic, fee, or general.
Do not provide any explanation.
"""
    
    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general" 
    return {'query_type' : category }


def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunk from the academic  handbook"""
    query = state["messages"][-1].content

    docs = academic_retriever.invoke(query)

    print("\nDEBUG — Academic query:", query)
    print("DEBUG — Retrieved chunks:", len(docs))

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    print("DEBUG — Context:\n", context)

    return {"retrieved_context": context}


def fee_rag_node(state : State) -> dict:
    """Retrieves relevant chunk from the fee structure handbook"""
    query = state["messages"][-1].content
    docs = fee_retrievr.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context" : context}


def general_node(state : State) -> dict:
    """Answer directly using LLM's own knowledge, no retrieval needed"""
    return {"retrieved_context" : "NO_RETRIEVAL_NEEDED"}



def response_node(state: State) -> dict:
    """Generate an answer using the retrieved PDF context."""

    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    query_type = state.get("query_type", "general")
    context = state.get("retrieved_context", "")

    if query_type == "general":
        prompt = f"""
        You are a friendly college assistant.

        Student's programme: {programme}
        Student's question: {query}

        Answer clearly and politely.
        Do not invent college-specific facts.
        If you do not know something, say so honestly.
        """
    else:
        prompt = f"""
        You are a college admission assistant.

        Student's programme: {programme}
        Question category: {query_type}
        Student's question: {query}

        Use ONLY the following retrieved PDF context
        to answer the student's question.

        PDF CONTEXT:
        {context}

        Instructions:
        1. Answer the exact question asked.
        2. Use relevant information from the context.
        3. If the question asks for a specific fee, amount,
           date, eligibility requirement, or programme detail,
           provide that exact information if available.
        4. Do not invent or guess missing information.
        5. If the context does not contain the answer,
           say that the available document does not provide it.
        6. Keep the answer clear and concise.
        """

    response = llm.invoke(prompt)

    return {
        "messages": [("ai", response.content.strip())]
    }



def route_query(state: State):
    if state["query_type"] == "academic":
        return "academic_rag"
    elif state["query_type"] == "fee":
        return "fee_rag"
    else:
        return "general"


##Graph 
graph = StateGraph(State)

graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag", academic_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

##edges

graph.add_edge(START,"classifier")
graph.add_conditional_edges(
    "classifier", route_query
)
graph.add_edge("academic_rag", "response")
graph.add_edge("fee_rag", "response")
graph.add_edge("general", "response")
graph.add_edge("response", END)

##Compile
app = graph.compile()

print("WElcome to the college assiatnt \n\n")
print("1 for BCA")
print("1 for BBA")
print("1 for B.COM")

choice = input('\nEnter 1,2 or 3')

programme_map = {
    "1":"BCA",
    "2":"BBA",
    "3":"B.COM"
}
student_programe = programme_map.get(choice, "BCA")
print(f"you are set as a {student_programe} student")

while True: 
    user_query = input("you: ")
    if user_query.lower() in ["exit","quit"]:
        break
    result = app.invoke({
         "programme":student_programe,
         "messages":[("human",user_query)]
     })
    print(f"Assistant:{result['messages'][-1].content}")
