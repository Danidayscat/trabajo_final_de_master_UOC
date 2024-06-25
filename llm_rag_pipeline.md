## Install

```python
# Basic data manipulation
!pip install -qU pandas numpy

# Machine learning and NLP
!pip install -qU scikit-learn nltk torch transformers

# Web scraping and HTML parsing
!pip install -qU requests beautifulsoup4

# Regular expressions
!pip install -qU regex

# Indexing and search
!pip install -qU llama-index
```

## Import

```python
import pandas as pd
import torch
import re
import numpy as np
from collections import defaultdict
from typing import List
from bs4 import BeautifulSoup
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.indices.service_context import ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (Settings, Document)
from llama_index.core.schema import MetadataMode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.query_engine import RetrieverQueryEngine
```

## Preprocessing

```python
# Directorio
csv_path= "sample-location" #radiology reports location

# Leer .csv
df = pd.read_csv(csv_path, encoding='ISO-8859-1', delimiter=';')

# Normalizar texto html
df['Contingut'] = df['Contingut'].apply(lambda x: BeautifulSoup(str(x).lower(), "html.parser").get_text() if isinstance(x, str) and '<' in x and '>' in x else x)

# Normalizar prueba diagnostica
df['Prova'] = df['Prova'].apply(lambda x: re.sub(r'\d+\.?', '', str(x)).replace('[', '').replace(']', '').strip().lower() if isinstance(x, str) else str(x).lower())

# Eliminar datos personales
def remove_data(text):
    if isinstance(text, str):
        return re.sub('barcelona.*', 'barcelona', text)
    return text

df['Contingut']=df['Contingut'].apply(remove_data)

# Columnas son numéricas
df['Historia'] = pd.to_numeric(df['Historia'], errors='coerce')
df['Exploracio'] = pd.to_numeric(df['Exploracio'], errors='coerce')
```
```python
# Verificar si hay duplicados
duplicados = df.duplicated(subset=["Historia", "Exploracio"], keep=False)

# Eliminar duplicados
df_dedup = df.drop_duplicates(subset=["Historia", "Exploracio"])

# Comparar tamaños antes y después
df_before = len(df)
df_after = len(df_dedup)

print(f'Antes: {df_before}')
print(f'Después: {df_after}')
```
```python
# Crear Document
documents = []

for idx, row in df_dedup.iterrows():
    document_id = row["Exploracio"]

    metadata = {
        "Numero exploracion": row["Exploracio"],
        "Numero historia": row["Historia"],
        "Fecha de consulta": row["Data"],
        "Prueba diagnostica": row["Prova"],
    }
    document = Document(
        text=row["Contingut"],
        id_=document_id,
        metadata=metadata,
        excluded_llm_metadata_keys=['file_path', 'file_type', 'file_size', 'creation_date', 'last_modified_date'],
        metadata_separator="::",
        metadata_template="{key}=>{value}",
        text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
    )
    
    documents.append(document)

print(f"Total: {len(documents)}")
```
```python
for i, doc in enumerate(documents[:2]):
    print(f"Document {i} ID: {doc.id_}")
    print("The LLM see this: \n", doc.get_content(metadata_mode=MetadataMode.LLM))
    #print("The embedding model see this: \n", doc.get_content(metadata_mode=MetadataMode.EMBED))
```
```
Document 0 ID: 3721809
The LLM see this: 
 Metadata: Numero exploracion=>3721809
Numero historia=>109368.0
Fecha de consulta=>03/01/2018
Prueba diagnostica=>mamografia
-----
Content:  mamografía.motivo de exploración: control ginecológico.resultados:mamas simétricas con predominio del tejido graso.no se observan lesiones focales sospechosas. conclusión: negativo. birads 1.densidad mamaria: a.
Document 1 ID: 3540069
The LLM see this: 
 Metadata: Numero exploracion=>3540069
Numero historia=>193146.0
Fecha de consulta=>03/01/2018
Prueba diagnostica=>mamografia
-----
Content: mamografía.motivo de exploración: control oncológico. antecedente de tumorectomía bilateral.resultados:mamas con tejido fibroepitelial denso de forma heterogénea. se observan cambios secundarios a tratamiento quirúrgico conservador y radioterapia de ambas mamas. la cicatriz de mi es un tanto densa, pero sin cambios significativos respecto a controles previos y en la cicatriz de md se observan calcificaciones de liponecrosis.proyectado sobre prolongación axilar izqueirda, a 12cm del pezón, se observa un nodulillo de bordes poco nítidos y 3mm de diámetro que parece ser de reciente aparición.no se observan lesiones focales sospechosas. conclusión: imagen en mi que requiere estudio complementario. birads 0. se programará mamografía con contraste y ecografía complementarias para su estudio más detallado.densidad mamaria: c.
```
```python
# Definir apartados
def extract_conclusion(text):
    conclusion_pattern = r'\b(?:conclusi[óo]ne?s?|impresi[óo]ne?s?(\s+diagn[oó]stica)?)\s*:\s*(.*)'
    match = re.search(conclusion_pattern, text, re.DOTALL)
    if match:
        return match.group(2).strip()
    return ""
```
```python
# Función para limpiar el texto de caracteres especiales
def clean_text(text):
    # Definir la expresión regular para permitir letras, números, espacios, puntuación y tildes
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,;:!?áéíóúÁÉÍÓÚñÑ]', '', text)
    return cleaned_text

# Definir función para extraer conclusiones
def extract_conclusion(text):
    text = clean_text(text)  # Limpiar el texto antes de extraer la conclusión
    conclusion_pattern = r'\b(?:conclusi[oó]ne?s?|impresi[oó]ne?s?(\s+diagn[oó]stica)?)\s*:\s*(.*)'
    match = re.search(conclusion_pattern, text, re.DOTALL)
    if match:
        return match.group(2).strip()
    return ""

# Extraer conclusiones y almacenar en una lista
conclusions = [extract_conclusion(doc.text) for doc in documents]

# Crear un DataFrame a partir de las conclusiones
df_conclusions = pd.DataFrame(conclusions, columns=['Conclusion'])

# Verificar y contar los duplicados
duplicated_conclusions = df_conclusions.duplicated(keep=False)
duplicated_df = df_conclusions[duplicated_conclusions]
duplicated_count = duplicated_df['Conclusion'].value_counts()

print("Conclusiones duplicadas y su conteo:")
print(duplicated_count)
```
```
Conclusiones duplicadas y su conteo:
Conclusion
                                                                                                                                                                                              2115
negativo. birads 1.densidad mamaria: b.                                                                                                                                                        543
negativo. birads 1.densidad mamaria: c.                                                                                                                                                        449
lesiones benignas. birads 2.densidad mamaria: b.                                                                                                                                               280
lesiones benignas. birads 2.densidad mamaria: c.                                                                                                                                               240
                                                                                                                                                                                              ... 
negativo. birads 1. controles habituales.densidad mamaria:b.                                                                                                                                     2
cambios secundarios a tratamiento conservador md. birads 2. controles periodicos.densidad: b.cambios secundarios a tratamiento conservador md. birads 2. controles periodicos.densidad: b.       2
negativo. birads 1.densidad mamaria c. controles clínicos ginecológicos                                                                                                                          2
birads 1 estudio dentro de la normalidad. se recomiendan controles periódicos habituales.densidad mamaria: b.                                                                                    2
birads 1 hallazgos dentro de la normalidad. densidad b. se recomiendan controles periódicos habituales.                                                                                          2
Name: count, Length: 1063, dtype: int64
```

```python
# Eliminar duplicados
df_conclusions_unique = df_conclusions.drop_duplicates()

# Contar conclusiones antes y después de eliminar duplicados
conclusions_before = len(df_conclusions)
conclusions_after = len(df_conclusions_unique)

print(f"Total de conclusiones antes de eliminar duplicados: {conclusions_before}")
print(f"Total de conclusiones después de eliminar duplicados: {conclusions_after}")
```
```
Total de conclusiones antes de eliminar duplicados: 17164
Total de conclusiones después de eliminar duplicados: 7296
```
```python
# Extraccion de BIRADS en Conclusion
def extract_bi_rads(text):
    bi_rads_matches = re.findall(r'bi\s*-?\s*rads\s*:? ?(4\s*[a-c]?|6|5|3|2|1|0)', text)
    if not bi_rads_matches:
        return 'no especifica BIRADS en el documento'
    
    bi_rads_norm = set()
    for match in bi_rads_matches:
        match = match.replace(" ","")
        if match.startswith('4'):
            if len(match) > 1:
                match = '4' + match[1].lower()
        bi_rads_norm.add(match)
    bi_rads = ", ".join(f"BIRADS {item}" for item in sorted(bi_rads_norm)) # Concatena
    return bi_rads

# Extraer lateralidad en Conclusion
def extract_lateralidad(text):
    lateralidad_matches = r'\b(derecha|izquierda|esquerra|dreta|md|mi)\b'
    matches = re.findall(lateralidad_matches, text)
    if not matches:
        return "no especifica lateralidad de la mama"
    
    lateralidad_norm = set()
    for match in matches:
        if match in ['derecha', 'dreta', 'md']:
            lateralidad_norm.add('mama derecha')
            text = re.sub(r'\b' + re.escape(match) + r'\b', "mama derecha", text)
        elif match in ['izquierda', 'esquerra', 'mi']:
            lateralidad_norm.add('mama izquierda')
            text = re.sub(r'\b' + re.escape(match) + r'\b', "mama izquierda", text)
        
    return ", ".join(sorted(lateralidad_norm)) if lateralidad_norm else "no especifica lateralidad de la mama"
```
```python
# Aplicar funciones extract_bi_rads y extract_lateralidad a las conclusiones únicas
bi_rads_results = df_conclusions_unique['Conclusion'].apply(extract_bi_rads)
lateralidad_results = df_conclusions_unique['Conclusion'].apply(extract_lateralidad)

# Crear un diccionario para mapear conclusiones a los resultados de las funciones
bi_rads_dict = dict(zip(df_conclusions_unique['Conclusion'], bi_rads_results))
lateralidad_dict = dict(zip(df_conclusions_unique['Conclusion'], lateralidad_results))

# Actualizar metadatos
for doc, conclusion_text in zip(documents, conclusions):
    doc.metadata["BI-RADS"] = bi_rads_dict[conclusion_text]
    doc.metadata["Lateralidad mama"] = lateralidad_dict[conclusion_text]

for doc in documents[:2]:
    print("The LLM see this: \n", doc.get_content(metadata_mode=MetadataMode.LLM))
    print("The embedding model see this: \n", doc.get_content(metadata_mode=MetadataMode.EMBED))
```
```
The LLM see this: 
 Metadata: Numero exploracion=>3721809
Numero historia=>109368.0
Fecha de consulta=>03/01/2018
Prueba diagnostica=>mamografia
BI-RADS=>BIRADS 1
Lateralidad mama=>no especifica lateralidad de la mama
-----
Content:  mamografía.motivo de exploración: control ginecológico.resultados:mamas simétricas con predominio del tejido graso.no se observan lesiones focales sospechosas. conclusión: negativo. birads 1.densidad mamaria: a.
The embedding model see this: 
 Metadata: Numero exploracion=>3721809
Numero historia=>109368.0
Fecha de consulta=>03/01/2018
Prueba diagnostica=>mamografia
BI-RADS=>BIRADS 1
Lateralidad mama=>no especifica lateralidad de la mama
-----
Content:  mamografía.motivo de exploración: control ginecológico.resultados:mamas simétricas con predominio del tejido graso.no se observan lesiones focales sospechosas. conclusión: negativo. birads 1.densidad mamaria: a.
```

## Transformation

```python
def homogenizer(text):
    text = re.sub(r'\bbi\s*-?\s*rads\s*:? ?', 'birads ', text)
    text = re.sub(r'\b4\s*(a|b|c)\b', r'4\1', text)
    text = re.sub(r'mama\s*derecha|mama\s*dreta|\bmd\b', "mama derecha", text)
    text = re.sub(r'mama\s*izquierda|mama\s*esquerra|\bmi\b', "mama izquierda", text)
    return text

def clean_puntuation(text):
    text = re.sub(r'[^A-Za-z0-9\s]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def eliminate_tildes(text, remove_tildes= True) -> str:
    # Convertir a minúsculas
    text = text.lower()
    if remove_tildes:
        # Reemplazar tildes agudas y graves por vocales base
        text = re.sub('[áà]', 'a', text)
        text = re.sub('[éè]', 'e', text)
        text = re.sub('[íì]', 'i', text)
        text = re.sub('[óò]', 'o', text)
        text = re.sub('[úù]', 'u', text)
    
    # Eliminar espacios adicionales
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Splitting y eliminacion de caracteres
patterns = {
        'Motivo': r'\b(ecografia|mamografia*motivo|mamografia\smotivo|mamografias*ecografia|mamografia\secografia)\b', 
        'Resultados': r'\b(resultado|resultados)\b', 
        'Conclusion': r'\b(conclusion|conclusiones|impresion|impresiones|impresiones+diagnostica)\b'
    }

def custom_sentence_splitter(text, chunk_size=None):
    sentences = []
    combined_pattern = '|'.join(patterns.values())
    matches = list(re.finditer(combined_pattern, text))

    if not matches:
        chunk_size = chunk_size or 500
        for i in range(0, len(text), chunk_size):
            sentences.append(text[i:i+chunk_size].strip())
        return sentences

    prev_end = 0
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if prev_end < start:
            sentences.append(text[prev_end:start].strip())
        sentences.append(text[start:end].strip())
        prev_end = end
    
    sentences = [s for s in sentences if s]
    return sentences
    
class CustomSentenceSplitter(SentenceSplitter):
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Override to use custom sentence.
        """
        return custom_sentence_splitter(text, chunk_size)

# Limpieza
for doc in documents:
    doc.text = homogenizer(doc.text)
    doc.text = eliminate_tildes(doc.text)
    doc.text = clean_puntuation(doc.text)

for doc in documents[:20]:
    print("The LLM see this: \n", doc.get_content(metadata_mode=MetadataMode.LLM))
```
## LLM Integration
```python
cache_folder = "./model_cache"
#embed_model = HuggingFaceEmbedding(
#    model_name="BAAI/bge-small-en", cache_folder = cache_folder)
embed_model = resolve_embed_model("local:BAAI/bge-small-en")

llm='local:mistral-7b-instruct-v0.1.Q4_K_M.gguf'
Settings.llm=llm
Settings.embed_model=embed_model
```
## Embedding Model

```python
def get_text_embedding(text):
    with torch.no_grad():
        embedding = embed_model(text.to(device))
    return embedding.cpu().numpy()

for doc in documents:
    embedding = embed_model.get_text_embedding(doc.get_content())
    doc.embedding = embedding

docstore = SimpleDocumentStore()
docstore.add_documents(documents)

for doc in documents[:5]:
    print(f"Document: {doc.id_}, Emb: {doc.embedding[:10]}...")
```
## Vector Storage

```python
def build_nodes(documents):
    nodes = []
    splitter = CustomSentenceSplitter(chunking_tokenizer_fn=None)

    for idx, doc in enumerate(tqdm(documents)):
        cur_nodes = splitter.get_nodes_from_documents([doc])
        for cur_node in cur_nodes:
            # ID will be base + parent
            file_name = doc.metadata["Numero exploracion"]
            new_node = IndexNode(
                text=cur_node.text or "None",
                index_id=str(file_name),
                metadata=doc.metadata,
            )
            nodes.append(new_node)
            
    print("num nodes: " + str(len(nodes)))
    return nodes

nodes = build_nodes(documents)

expected_num_nodes = len(documents)
actual_num_nodes = len(nodes)

print(f"Extected: {expected_num_nodes}")
print(f"Actual: {actual_num_nodes}")

for node in nodes[:5]:
    print(f"Node ID: {node.index_id}, Metadata: {node.metadata}")
```
```python
def save_index(nodes, embed_model, out_path):
    out_path = os.path.abspath(out_path)
    #print(f"Ruta {out_path}")

    #if not os.access(out_path, os.W_OK):
        #print("No hay permiso")
        #return None
    
    if not os.path.exists(out_path):
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        index.set_index_id("simple_index")
        index.storage_context.persist(f"./{out_path}")
    else:
        print("Directorio si existe")
        # rebuild storage context
        storage_context = StorageContext.from_defaults(
            persist_dir=out_path
        )
        # load index
        index = load_index_from_storage(
            storage_context, index_id="simple_index", embed_model=embed_model
        )

    return index

# Guardar índice
out_path = "./chunck_index"
index = save_index(nodes, embed_model, out_path)
```

## Retrieval Agent
```python
lass HybridRetriever(BaseRetriever):
    """Hybrid retriever."""

    def __init__(
        self,
        vector_index,
        docstore,
        similarity_top_k: int = 2,
        out_top_k: Optional[int] = None,
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(**kwargs)
        self._vector_index = vector_index
        self._embed_model = vector_index._embed_model
        self._retriever = vector_index.as_retriever(
            similarity_top_k=similarity_top_k
        )
        self._out_top_k = out_top_k or similarity_top_k
        self._docstore = docstore
        self._alpha = alpha

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        # first retrieve chunks
        nodes = self._retriever.retrieve(query_bundle.query_str)

        # get documents, and embedding similiaryt between query and documents

        ## get doc embeddings
        docs = [self._docstore.get_document(n.node.index_id) for n in nodes]
        doc_embeddings = [d.embedding for d in docs]
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )

        ## compute doc similarities
        doc_similarities, doc_idxs = get_top_k_embeddings(
            query_embedding, doc_embeddings
        )

        ## compute final similarity with doc similarities and original node similarity
        result_tups = []
        for doc_idx, doc_similarity in zip(doc_idxs, doc_similarities):
            node = nodes[doc_idx]
            # weight alpha * node similarity + (1-alpha) * doc similarity
            full_similarity = (self._alpha * node.score) + (
                (1 - self._alpha) * doc_similarity
            )
            full_similarity = round(full_similarity, 2)
            print(
                f"Doc {doc_idx} (node score, doc similarity, full similarity): {(round(node.score, 2), round(doc_similarity, 2), full_similarity)}"
            )
            result_tups.append((full_similarity, node))

        result_tups = sorted(result_tups, key=lambda x: x[0], reverse=True)
        
        # update scores
        for full_score, node in result_tups:
            node.score = full_score

        return [n for _, n in result_tups][:out_top_k]
```
```python
top_k = 10
out_top_k = 3
hybrid_retriever = HybridRetriever(
    index, docstore, similarity_top_k=top_k, out_top_k=3, alpha=0.5
)

base_retriever = index.as_retriever(similarity_top_k=out_top_k)
```
```python
def show_nodes(nodes, out_len: int = 5000):
    for idx, n in enumerate(nodes):
        lateralidad = n.metadata.get('Lateralidad mama')
        bi_rads = n.metadata.get('BI-RADS')
        print(f"\n\n >> ID, Lateralidad: {lateralidad}, BI-RADS: {bi_rads}")
        print(n.get_content()[:out_len])
``
```python
query_str = "Muestra numero de informes con birads 5 en la mama izquierda"
nodes = hybrid_retriever.retrieve(query_str)
show_nodes(nodes)
```

## Preguntas
```python
custom_query_engine = RetrieverQueryEngine(hybrid_retriever)
questions= ["¿Cuál es el birads para esta frase: lesiones benignas?",
            "¿Cuál es el birads para esta frase: exploracion incompleta, exploracion insuficiente o valoracion adicional?", 
            "¿Cuál es el birads para esta frase: probablemente benignas?",
            "¿Cuál es el birads para esta frase: alta sospecha de malignidad?",
            "¿Cuál es el birads para esta frase: baja sospecha de malignidad?",
            "¿Cuál es el birads para esta frase: moderada sospecha de malignidad?",
            "¿Cuál es el birads para esta frase: altamente sospechosa de malignidad?",
            "¿Cuál es el birads para esta frase: realización de biopsia, realización de marcaje o neoplasias?"]

responses = {}

for question_str in questions:
    print(f"Pregunta: {question_str}")
    response_str = custom_query_engine.query(question_str)
    responses[question_str] = response_str
    print(f"Respuesta: {str(response_str)}\n")
```
