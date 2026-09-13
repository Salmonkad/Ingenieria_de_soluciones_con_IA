import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    except Exception:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
else:
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def simular_ingesta_fuentes_institucionales() -> List[Document]:
    return [
        Document(
            page_content=(
                "Reglamento de Beneficios y Becas DAE - Artículo 12: Para conservar la Beca de "
                "Excelencia Académica Institucional, el estudiante regular debe mantener un promedio "
                "ponderado acumulado semestral igual o superior a 5.5, no registrar sanciones "
                "disciplinarias y presentar un avance curricular mínimo del 80% de asignaturas inscritas."
            ),
            metadata={"origen": "interno", "tipo": "normativo", "fuente": "Reglamento_Becas_DAE_2026.pdf"}
        ),
        Document(
            page_content=(
                "Decreto Exento MINEDUC N° 450 - Gratuidad y Beneficios Estatales: La renovación "
                "de la gratuidad requiere matrícula anual en los plazos fijados por la institución adscrita "
                "y permanecer dentro de la duración formal (nominal) de la carrera informada en el SIES."
            ),
            metadata={"origen": "externo", "tipo": "decreto_ley", "fuente": "Decreto_MINEDUC_Gratuidad.pdf"}
        ),
        Document(
            page_content=(
                "Calendario Académico 2026: "
                "Hito 1: Plazo límite de matrícula para continuidad de estudios: 20 de marzo de 2026. "
                "Hito 2: Periodo de apelación a pérdidas de beneficios DAE: del 02 al 15 de marzo de 2026. "
                "Hito 3: Publicación de resultados de asignación y renovación: 31 de marzo de 2026."
            ),
            metadata={"origen": "interno", "tipo": "calendario_academico", "fuente": "Calendario_Oficial_2026.csv"}
        ),
    ]


def inicializar_vectorstore():
    documentos = simular_ingesta_fuentes_institucionales()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_documents(documentos)
    return Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name="dae_knowledge_base")


SYSTEM_PROMPT = """Eres 'Asesores de DAE', asistente virtual formal, empático y riguroso de la Dirección de Asuntos Estudiantiles.
Tu única labor es orientar a los estudiantes sobre continuidad de estudios y normativas de becas.

REGLAS OBLIGATORIAS:
1. Responde ÚNICA y EXCLUSIVAMENTE basándote en el contexto suministrado.
2. Si el contexto NO contiene la información sobre fechas o requisitos, responde obligatoriamente:
   "Para darte una respuesta certera sobre tu caso particular, por favor acércate a la oficina DAE presencial o levanta un ticket en el portal."
3. NUNCA inventes promedios mínimos, notas de aprobación, porcentajes ni fechas.
4. Incluye siempre una recomendación explícita para que el alumno revise permanentemente su correo institucional.
5. Cita explícitamente la fuente de la información (ejemplo: 'Según el Reglamento de Becas DAE...', 'Conforme al Decreto MINEDUC...').

Contexto recuperado:
{context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])


class AgenteDAE:
    def __init__(self):
        if not os.getenv("LLM_API_KEY"):
            raise EnvironmentError("Falta la variable de entorno LLM_API_KEY")

        self.vectorstore = inicializar_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        self.llm = ChatOpenAI(
            base_url=os.getenv("LLM_BASE_URL", "https://api.mistral.ai/v1"),
            api_key=os.getenv("LLM_API_KEY"),
            model=os.getenv("LLM_MODEL_NAME", "mistral-small-latest"),
            temperature=0.0
        )
        self.prompt_template = prompt_template

    @staticmethod
    def _formatear_contexto(docs: List[Document]) -> str:
        return "\n\n".join([f"[{d.metadata.get('fuente')}] {d.page_content}" for d in docs])

    def consultar(self, pregunta: str) -> Dict[str, Any]:
        docs_recuperados = self.retriever.invoke(pregunta)
        contexto = self._formatear_contexto(docs_recuperados)
        mensajes = self.prompt_template.format_messages(context=contexto, question=pregunta)
        respuesta = self.llm.invoke(mensajes).content

        return {
            "pregunta": pregunta,
            "respuesta": respuesta,
            "evidencias": [
                {
                    "fuente": d.metadata.get("fuente"),
                    "tipo": d.metadata.get("tipo"),
                    "origen": d.metadata.get("origen"),
                    "extracto": d.page_content[:120] + "..."
                }
                for d in docs_recuperados
            ]
        }


if __name__ == "__main__":
    agente = AgenteDAE()

    casos_prueba = [
        "¿Qué promedio necesito para mantener mi Beca de Excelencia Académica?",
        "¿Cuál es el último día para matricularme según el calendario?",
        "¿Cuánto saldo me queda en la tarjeta JUNAEB este mes?"
    ]

    for i, consulta in enumerate(casos_prueba, start=1):
        resultado = agente.consultar(consulta)
        print(f"\n[Test Case {i}] Pregunta: {resultado['pregunta']}")
        print(f"-> Respuesta Agente:\n{resultado['respuesta']}\n")
        print("-> Trazabilidad de Documentos Recuperados:")
        for ev in resultado["evidencias"]:
            print(f"   * [{ev['origen'].upper()}] {ev['fuente']} | Tipo: {ev['tipo']}")
        print("-" * 80)
