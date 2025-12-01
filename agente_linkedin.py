from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.types import Command, interrupt
from langchain_core.tools import tool
import sys
from langchain_core.messages import HumanMessage, SystemMessage
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# memoria

config = {"configurable": {"thread_id": "1"}}

checkpointer = MemorySaver()
# checkpointer = SqliteSaver(conn=sqlite3.connect("agente.db", check_same_thread=False))

# tools

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

@tool
def linkedin_search(search_terms: str) -> str:
    """
    Search for LinkedIn profiles using specific industry terms, job titles, or company types.
    
    Args:
        search_terms: Specific keywords, job titles, industries, or company types to search for
                     (e.g., "restaurant industry CEO", "pet services entrepreneur", "fintech startup founder")
    """
    tavily = TavilySearch(max_results=5)
    
    # Search for both individual profiles and companies
    linkedin_query = f"site:linkedin.com/in {search_terms} OR site:linkedin.com/company {search_terms}"
    
    try:
        # Get results from Tavily
        search_results = tavily.run(linkedin_query)
        
        # Handle different response formats from Tavily
        if isinstance(search_results, str):
            # If Tavily returns a string, parse it or return as is
            return f"LinkedIn search results for '{search_terms}':\n\n{search_results}"
        
        if isinstance(search_results, list):
            results = search_results
        elif isinstance(search_results, dict) and 'results' in search_results:
            results = search_results['results']
        else:
            return f"Unexpected response format from search for: {search_terms}"
        
        if not results:
            return f"No LinkedIn profiles found for: {search_terms}"
        
        formatted_results = f"LinkedIn profiles found for '{search_terms}':\n\n"
        
        # Process up to 5 results
        for i, result in enumerate(results[:5], 1):
            if isinstance(result, dict):
                url = result.get('url', 'No URL')
                title = result.get('title', 'No title')
                content = result.get('content', result.get('snippet', 'No description available'))
            else:
                # If result is not a dict, convert to string
                url = 'No URL'
                title = f'Result {i}'
                content = str(result)
            
            # Clean up the content to make it more readable
            content_preview = content[:200] + "..." if len(content) > 200 else content
            
            formatted_results += f"{i}. **{title}**\n"
            formatted_results += f"   URL: {url}\n"
            formatted_results += f"   Description: {content_preview}\n\n"
        
        return formatted_results
        
    except Exception as e:
        return f"Error searching LinkedIn profiles: {str(e)}"

tools = [
    human_assistance,
    linkedin_search,
]

# llm

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

llm_with_tools = llm.bind_tools(tools)

# grafo

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent1(state: State):
    system_prompt = """
Eres un agente especializado en encontrar perfiles junior del sector informático en LinkedIn, con no más de 3 años de experiencia laboral. Tu tarea es:
1. Recibir una descripción del tipo de perfil que el usuario necesita
2. Analizarla para identificar roles y tipos de empresas relevantes
3. Generar consultas de búsqueda específicas y accionables para LinkedIn que encuentren personas de Argentina interesadas en moverse a España

Cuando recibas una descripción de perfil:

PASO 1: Analiza el perfil objetivo e identifica:
- ¿Qué roles/posiciones junior son relevantes? (ej.: "Junior Software Developer", "Junior Data Analyst", "Junior QA Engineer")
- ¿Qué tecnologías, lenguajes o áreas indican que es sector informático? (ej.: "Python", "Java", "Front-end", "Ciberseguridad", "DevOps")
- ¿Qué tipo de empresas son adecuadas? (ej.: consultoras IT, startups tecnológicas, empresas de producto digital, hubs tecnológicos)
- ¿Qué señales indican interés en movilidad hacia España? 
  (ej.: "open to relocation", "relocation to Spain", "mudanza a España", "trabajar en España", "open to work Spain", "Europe", "UE")

PASO 2: Ten en cuenta el foco geográfico:
- Perfiles que actualmente estén en Argentina (ej.: "Argentina", "Buenos Aires", "Córdoba", "Rosario", etc.)
- Interés explícito o implícito en moverse a España
- Usa combinaciones de términos como:
  - "Argentina" AND "relocation Spain"
  - "Argentina" AND "interesado en trabajar en España"

PASO 3: Forma consultas de búsqueda específicas y orientadas a LinkedIn:
- En lugar de usar descripciones largas, crea combinaciones de:
  * Rol/posición junior
  * Tecnología/área (opcional)
  * Términos de movilidad o interés en España
  * Referencia clara a Argentina como ubicación actual

Ejemplos de consultas:
- "Junior Software Developer Argentina open to relocation Spain"
- "Junior Data Engineer Argentina trabajar en España"
- "Junior Python developer Argentina interested in moving to Spain"

No uses ninguna herramienta externa, sólo responde con las consultas de búsqueda que has generado y, opcionalmente, una breve explicación de a qué tipo de perfil apunta cada consulta.
    """
    conversation = [
        SystemMessage(content=system_prompt),
        *state["messages"]
    ]
    return {"messages": [llm_with_tools.invoke(conversation)]}

def agent2(state: State):
    system_prompt = """
        Eres un agente especializado en encontrar y pre-qualificar perfiles de informática para contratación usando LinkedIn. Tu tarea es:
        1. Recibir consultas de búsqueda específicas de LinkedIn orientadas a contratar perfiles informáticos
        2. Buscar perfiles de LinkedIn relevantes para esas consultas
        3. Devolver un resumen estructurado de los perfiles encontrados junto con mensajes de contacto y preguntas clave para evaluar el encaje


        Cuando recibas una consulta de búsqueda específica de linkedin:

        PASO 1: Usa linkedin_search con los términos específicos que identificaste 

        PASO 2: Para cada perfil encontrado, devuelve la siguiente información:
        - Nombre completo
        - Titular de LinkedIn (headline)
        - Ubicación
        - Rol/posición actual (si está disponible)
        - Tecnologías/claves técnicas mencionadas (ej.: Python, Java, React, AWS, Data, QA, etc.)
        - Nivel estimado (junior / mid / senior) en base a años de experiencia y puestos anteriores
        - Enlace al perfil (si está disponible en la respuesta de la herramienta)

        PASO 3: Para cada perfil encontrado, crea:
        - Un mensaje de contacto personalizado orientado a reclutamiento/colaboración que:
          * Mencione el rol o stack relevante del perfil
          * Destaque brevemente el tipo de oportunidad (ej.: "puesto junior en backend en Málaga, España", "oportunidad remota en Europa", etc.)
          * Sea breve, profesional y cercano
IMPORTANTE: Necesito perfiles reales de linkedin, no perfiles de empresas y/o personas inventadas.

        Ejemplo:
        Consulta: "Junior Python Developer Buenos Aires open to work"
        Resultado:
        - Perfil 1: "Ana García, Junior Python Developer | Data & Backend"
            - Ubicación: Buenos Aires, Argentina
            - Tecnologías: Python, Django, SQL, APIs REST
            - Nivel estimado: Junior
            - Mensaje de contacto:
              "Hola Ana, soy [Tu nombre] y trabajo en [Tu empresa]. Estamos buscando un/a Junior Python Developer en Málaga, para un proyecto de backend y data con Python y Django. He visto tu experiencia y encaja muy bien con lo que buscamos. 
              ¿Te interesaría que te cuente más detalles sobre el puesto y el equipo?"
            

        - Perfil 2: "Carlos López, Junior Backend Developer | Python | Flask"
            - Ubicación: Córdoba, Argentina
            - Tecnologías: Python, Flask, PostgreSQL, Docker
            - Nivel estimado: Junior
            - Mensaje de contacto:
              "Hola Carlos, soy [Tu nombre] y formo parte del equipo de selección en [Tu empresa]. Estamos ampliando nuestro equipo backend con perfiles junior que trabajen con Python y bases de datos relacionales. 
              He visto tu experiencia con Python y Flask y me parece muy alineada con lo que estamos construyendo. 
              ¿Te gustaría que te comparta más detalles sobre el proyecto y las condiciones?"

        Sé proactivo y ejecuta las herramientas necesarias sin esperar confirmación.
            """
    conversation = [
        SystemMessage(content=system_prompt),
        *state["messages"]
    ]
    return {"messages": [llm_with_tools.invoke(conversation)]}

tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(State)
graph_builder.add_edge(START, "agent1")
graph_builder.add_node("agent1", agent1)
graph_builder.add_node("agent2", agent2)
graph_builder.add_edge("agent1", "agent2")
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("agent2", tools_condition)
graph_builder.add_edge("tools", "agent2")
graph = graph_builder.compile(checkpointer=checkpointer)
with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

# ejecución

while True:
    profile_description = input("💡 Profile Description: ")
    if profile_description.lower() in ["quit", "exit", "q"]:
        print("Goodbye! 👋")
        break
    events = graph.stream(
        {"messages": [{"role": "user", "content": profile_description}]},
        config, # para seguir un hilo de conversación
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
    # Check if we need human input for a tool
    while True:
        snapshot = graph.get_state(config)
        if snapshot.next and snapshot.next[0] == 'tools':
            # Tool is waiting for human input
            human_response = input("Human: ")
            if human_response.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                sys.exit(0)
            # Resume the tool execution with human response
            human_command = Command(resume={"data": human_response})
            events = graph.stream(human_command, config, stream_mode="values")
            # Process the response after human input
            for event in events:
                if "messages" in event:
                    event["messages"][-1].pretty_print()
        else:
            # No more tools waiting, break out of the inner loop
            break