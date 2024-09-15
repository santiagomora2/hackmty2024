# Inspiration
According to The conference Board since the beginning of 2024, retail sales have been down . This is due to slower wage growth, depleted savings and rising credit card debt, amongst other factors. This is why brick and mortar stores need to adapt newer technologies to find ways to engage with their costumers. This is were Ojo de buen cubero comes in, it takes advantage of two uses of AI, GenAI as a shopping assistant and Computer Vision to help business owners figure out what products in their floor are attracting customers and where they need to put their attention, for this specific project we decided to set it in a fictional car dealership.

# What it does
The part designed for the business owner uses the YOLO algorithm to track the routes of the customers inside the "showroom", and designated interest zones based on the radius of the car, latter we used this to create a heatmap of the foot traffic in the showroom. For the user experience we trained a language model with information of cars to do a chat bot working as a shopping assistant.

# How we built it
The project was built using Python. The chat bot was based on the OpenAI API, but trained using Langchain, with vector databases processed from technical specification from manufacturers. The computer vision aspect was made mainly with the YOLO algorithm modeled with the help of OpenVINO. With this model we were able to register the paths the clients took and find the areas or products that might need more attention

# Challenges we ran into
We weren´t familiar with the frameworks the challenge required us to use. Also the quick changes in the documentation of Langchain made it hard to keep track and learn it.

# Accomplishments that we're proud of
We were able to deliver a project with tools we were unfamiliar with.

# What's next for Ojo de buen cubero
This project is scalable in manny ways, but we think a big one would be to connect the information given by computer vision with the chat bot, in order to recommend better products or personalize the conversation. We also think that it's posible to use this project for other kinds of small retail stores.
