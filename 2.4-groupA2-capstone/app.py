import random
import gradio as gr
import re

from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load LLM
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")

# Load Knowledge Base
with open("music_knowledge_base.txt", "r", encoding="utf-8") as f:
    knowledge_base = f.read()

# Split Knowledge Base into Chunks
chunks = [
    chunk.strip()
    for chunk in knowledge_base.split("==================================================")
    if len(chunk.strip()) > 50
]

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create Embeddings
chunk_embeddings = embedding_model.encode(
    chunks,
    convert_to_tensor=True
)

# Retrieve Relevant Context
def retrieve_context(query, top_k=8):

    query_embedding = embedding_model.encode(
        query,
        convert_to_tensor=True
    )

    similarities = cos_sim(query_embedding, chunk_embeddings)[0]

    top_indices = similarities.argsort(descending=True)[:top_k]

    context = "\n\n".join(
        chunks[i.item()] for i in top_indices
    )

    return context

def extract_song_recommendations(context, max_songs=10):
    recommendations = []
    sections = context.split("Artist:")

    for section in sections:
        if not section.strip():
            continue

        lines = section.splitlines()
        artist = lines[0].strip()
        songs = []
        collecting = False
        for line in lines:
            line = line.strip()
            if line.lower().startswith("popular songs"):
                collecting = True
                continue

            if collecting:
                if line == "":
                    break
                if line.startswith("-"):
                    songs.append(line[1:].strip())
        if songs:
            recommendations.append(
                (
                    artist,
                    random.choice(songs)
                )
            )

    random.shuffle(recommendations)

    return recommendations[:max_songs]

# Chatbot
def respond(message, history):

    context = retrieve_context(message)

    print("\n========== RETRIEVED CONTEXT ==========")
    print(context)
    print("=======================================\n")

    messages = [
        {
            "role": "system",
            "content": f"""
You are MatchMyMusic, an AI music recommendation chatbot.
Use ONLY the retrieved context below.
IMPORTANT RULES:
1. When a user ask for recommendations, first EXPLICITLY ASK THE USER FOR their taste preference before giving recommendations.
2. Every artist you recommend MUST appear in the retrieved context.
3. Every song you recommend MUST appear under that SAME artist in the retrieved context.
4. Never invent:
- songs
- albums
- playlists
- artist/song combinations
5. Never move songs between artists.
6. If the retrieved context does not contain enough information, say:
"I don't have enough information in my knowledge base."
7. Double-check every recommendation before responding.
8. Explain WHY each recommendation matches the user's request.
Retrieved Context:
{context}
"""
        }
    ]

    if history:
        messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": message
        }
    )

    response = client.chat_completion(
        messages=messages,
        max_tokens=450,
        temperature=0.05
    )

    return response.choices[0].message.content.strip()
custom_theme = gr.themes.Citrus(
    primary_hue="yellow",
    neutral_hue="amber",
    spacing_size="lg",
    radius_size="lg",
    text_size="lg",
    font=["gstaff/sketch"],
).set(
    background_fill_primary="#FFF4C2",
    background_fill_secondary="#FFF9E6"
)

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🎶MatchMyMusic🎶
        Get your own personalized music recommendations based on your taste, mood, and more!
        """,
        elem_id="title"
    )

#with gr.Blocks() as chatbot:
    #gr.Image(
	    #value="piano.jpg", 
	    #show_label=False, 
	    #show_share_button = False, 
	    #show_download_button = False)
    #gr.ChatInterface(respond, type="messages")

    gr.ChatInterface(
        fn=respond,
        examples=[
            "I really like Lana Del Rey, what else should I listen to?",
            "Can you make me a chill playlist for studying?",
            "I'm feeling really happy, can you give me recommendations for energetic music?",
        ]
    )

demo.launch(ssr_mode=False, theme=custom_theme)