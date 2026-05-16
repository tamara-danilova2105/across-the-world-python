from fastapi import FastAPI
from routers.analysis import router as analysis_router
from exceptions import install_exception_handlers

# from playground.pipeline_sentiment import main as sentiment_debug
# from playground.pipeline_topics import main as topics_debug
# from playground.edge_cases_sentiment import main as edge_cases_debug
# from playground.manual_sentiment_inference import main as manual_sentiment_inference
# from playground.embeddings_similarity import main as embeddings_similarity
# from playground.semantic_search import main as semantic_search
# from playground.embeddings_edge_cases import main as embeddings_edge_cases

app = FastAPI(title="Review Analysis API", version="1.0.0")
install_exception_handlers(app)


app.include_router(analysis_router, prefix="/v1")

# TEMP DEBUG
# sentiment_debug()
# topics_debug()
# edge_cases_debug()
# manual_sentiment_inference()
# embeddings_similarity()
# semantic_search()
# embeddings_edge_cases()