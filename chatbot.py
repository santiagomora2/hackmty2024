# Librerías para procesamiento de mensajes y respuestas con IA

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate


# Librerías para chatbot de Telegram

import os
from typing import Final
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Cargar las variables de entorno del archivo .env
load_dotenv()

# Access the variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TOKEN: Final = os.getenv('TOKEN')

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

BOT_USERNAME: Final = '@gear_shift_bot'

# Definiciones para procesamiento de texto

# Cómo se convertirán preguntas a vectores / vectores a texto
embeddings = OpenAIEmbeddings()

#LLM
llm = ChatOpenAI(
    model = 'gpt-4o-mini',
    temperature=1
)

# Prompt
system_prompt = (
    'Actúa como un agente de ventas de volkswagen. Da respuestas con base en el contexto que se te da sin mencionar a otras marcas de autos. Tu objetivo es resaltar lo positivo de los autos y vendérlos a un cliente.'
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


index_name = False

""" COMMANDOS : 

En esta sección se definen los comandos del chatbot: que sucede cuando se inicializa (/start)
cuando se pide ayuda /ayuda o cuando se quiere información de un carro nuevo (/carronuevo)

"""

# /start
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Menú de opciones para que el cliente escoja los carros
    keyboard = [
        [InlineKeyboardButton("Nivus", callback_data='1')],
        [InlineKeyboardButton("T-Cross", callback_data='2')],
        [InlineKeyboardButton("Polo", callback_data='3')],
        [InlineKeyboardButton("Teramont", callback_data='4')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text('¡Hola! Soy GearShift, tu asistente para escoger tu auto ideal. ¿De qué coche deseas información?', reply_markup=reply_markup)

# Maneja las respuestas del menú
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    global index_name
    global vectorstore

    # Responde según la opción elegida
    if query.data == '1':
        index_name = 'nivus'
        await query.edit_message_text(text="¿Qué quieres saber sobre el VW Nivus?")
    elif query.data == '2':
        index_name = 'tcross'
        await query.edit_message_text(text="¿Qué quieres saber sobre el VW Nivus T-Cross?")
    elif query.data == '3':
        index_name = 'polo'
        await query.edit_message_text(text="¿Qué quieres saber sobre el VW Polo?")
    elif query.data == '4':
        index_name = 'teramont'
        await query.edit_message_text(text="¿Qué quieres saber sobre el VW Teramont?")

    #Manda llamar la base de datos vectorial para cada carro, para responder de acorde a esto
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# /ayuda
async def ayuda_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Si deseas obtener información de un auto, escribe /seleccionarcarro '+
                                    'y con gusto te brindaré ayuda.')

# /seleccionarcarro
async def seleccionar_carro_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # De nuevo, menú desplegable con opciones de carro a escoger
    keyboard = [
        [InlineKeyboardButton("Nivus", callback_data='1')],
        [InlineKeyboardButton("T-Cross", callback_data='2')],
        [InlineKeyboardButton("Polo", callback_data='3')],
        [InlineKeyboardButton("Teramont", callback_data='4')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text('¿De qué coche deseas información?', reply_markup=reply_markup)


"""
Lógica para las respuestas

En esta sección se define la lógica que se lleva a cabo cuando el bot recibe un mensaje.
En este caso, se contesta con LLM 
"""

def handle_response(text: str) -> str:

    # se convierte a minúsculas todo el texto
    pregunta: str = text.lower()

    # si no hay una base de datos vectorial asociada, no se contesta y se pide seleccionar un auto
    if not index_name:
        respuesta = 'No has seleccionado ningún carro! por favor ecribe /seleccionarcarro y escoge uno'
    else:
        # De lo contrario se genera la respuesta basandose en los datos existentes y en el LLM seleccionado

        qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type = "stuff",
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3}),
        )

        respuesta = qa_chain.invoke(pregunta)

    return respuesta['result']


# Cómo actúa el bot si es en mensaje personal o en Grupo
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    # Para debugging: imprime los mensajes en consola
    print(f'User {update.message.chat.id} in {message_type}: {text}')

    # Si es en grupo, sólo contesta si se le menciona
    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_response(new_text)
        else:
            return
    else:
        #Si es en privado, contesta siempre
        response: str = handle_response(text)

    print('Bot: ', response)
    await update.message.reply_text(response)

# Si hay un error, lo imprime en consola para facilitar debugging
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error: {context.error}')

"""
Main:

Aquí se construye la app con la TOKEN pertinente
"""

if __name__ == '__main__':
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('ayuda', ayuda_command))
    app.add_handler(CommandHandler('seleccionarcarro', seleccionar_carro_command))

    # Mensajes
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_handler(CallbackQueryHandler(button))

    # Errores
    app.add_error_handler(error)

    # Polling
    app.run_polling(allowed_updates=Update.ALL_TYPES)

