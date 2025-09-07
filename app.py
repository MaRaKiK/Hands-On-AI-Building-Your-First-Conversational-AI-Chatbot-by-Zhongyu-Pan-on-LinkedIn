#----------Import of required libraries into this notebook----------
import gradio as gr
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

#----------Checking for the GPU readiness environment for faster excecussion----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#----------Load pretrained DialoGPT (medium size) model and tokenizer----------
MODEL_NAME = "microsoft/DialoGPT-medium"
#Medium is chosen because it balances performance and compute requirements.
#AutoModelForCausalLM loads the model architecture + pretrained weights.
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
#Tokenizer converts text into tokens (numerical IDs).
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#----------Load dataset----------
from datasets import load_dataset
#DailyDialog failed to load so Blended Skill Talk is the one loaded 
try:
    dataset = load_dataset("daily_dialog")
    print("Loaded dataset: DailyDialog")
    dialog_column = "dialog"
except Exception as e:
    print("DailyDialog failed to load, switching to Blended Skill Talk...")
    dataset = load_dataset("blended_skill_talk")
    print("Loaded dataset: Blended Skill Talk")
    dialog_column = "context" # 'context' is the column containing dialogue in blended_skill_talk

#Reduce dataset size for quicker training (take ~1/30 of data, shuffled data for a fairer and more better training)
train_data = dataset["train"].shuffle(seed=42).select(range(len(dataset["train"]) // 30))
valid_data = dataset["validation"].shuffle(seed=42).select(range(len(dataset["validation"]) // 30))

#----------Tokenization----------
#Set padding token to EOS token (important for dialogue models)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    #Some datasets store dialogue as lists of turns so flatten them into one string
    text_list = [" ".join(dialog) if isinstance(dialog, list) else dialog for dialog in examples[dialog_column]]

    #Tokenize each conversation
    model_inputs = tokenizer(text_list, padding="max_length", truncation=True, max_length = 128)

    #Set labels into input_ids(standard for causal language modeling)
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs

#Apply tokenization to train & validation sets
tokenized_train = train_data.map(tokenize_function, batched=True, remove_columns=[dialog_column])
tokenized_valid = valid_data.map(tokenize_function, batched=True, remove_columns=[dialog_column])


#Convert dataset format into PyTorch tensors
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_valid.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

#----------Training setup----------
training_args = TrainingArguments(
    output_dir="./fine_tuned_chatbot",
    learning_rate=5e-5,                #Learning rate for optimizer (how fast the model updates its weights) 5e-5 is common default for fine-tuning transformers
    per_device_train_batch_size=2,     #How many samples are processing on before updating gradients, 2 sequences per step, small to avoid running out of memory
    per_device_eval_batch_size=2,      #Batch size used during evaluation (validation set)
    num_train_epochs=3,                #How many times the model goes through the whole dataset, 3 times the more the better
    save_steps=500,                    #Save model checkpoint every 500 training steps
    save_total_limit=2                 #Keep only the last 2 checkpoints to save disk space
)

trainer = Trainer(
    model=model,                       
    args=training_args,               #Configuration by initialize arguments
    train_dataset=tokenized_train,    #Tokenized training data
    eval_dataset=tokenized_valid      #Tokenized training data
)

#Train the model
trainer.train()
print("Training complete.")

#----------Chatbot's response----------
def chatbot_response(user_input):

    #Encode user input into tokens
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(model.device)
    #Generate a reply using sampling (adds randomness for more natural responses)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=30,                         #Limit reply length to 30 tokens
        pad_token_id=tokenizer.eos_token_id,       #Ensures padding is handled properly
        do_sample=True,                            #Enables sampling (so responses are more creative, not deterministic)
        top_k=50,                                  #Restricts sampling to the 50 most likely tokens (reduces randomness)
        top_p=0.9,                                 #Nucleus sampling, keeps the smallest set of tokens whose cumulative probability ≥ 0.9
        temperature=0.7,                           #Controls randomness (lower = safer, higher = more creative)
        repetition_penalty=1.2                     #Discourages the model from repeating itself
    )
    #Decode tokens back to text (skip special tokens like <EOS>)
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response



#----------Gradio UI for chatbot----------

css = """
/* Container */
.container {
    background-color: #b9cff9;
    border-radius: 10px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    padding: 25px;
    font-family: 'Roboto', sans-serif;
}
/* Title */
h1 {
    text-align: center;
    font-size: 28px;
    color: #6d9dc0;
    font-weight: 600;
    margin-bottom: 20px;
    font-family: 'Roboto', sans-serif;
}
/* Chatbot bubbles */
.gr-chatbot .message.user {
    background-color: ##fedefd !important;
    color: white !important;
    border-radius: 15px;
    padding: 10px 14px;
}
.gr-chatbot .message.bot {
    background-color: #cfe2ff !important;
    color: black !important;
    border-radius: 15px;
    padding: 10px 14px;
}
/* Input box */
input[type="text"] {
    padding: 12px 16px;
    border-radius: 25px;
    border: 2px solid #ff6f61;
    background-color: #fff9e6;
    color: brown;
    font-weight: bold;
}
/* Button */
button {
    background-color: #b9cff9 !important;
    color: white !important;
    padding: 12px 20px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 25px !important;
    border: none;
    cursor: pointer;
}
button:hover {
    background-color: ##81b0d2 !important;
    transform: scale(1.05);
}
"""

#Launch of gradio interface created
iface = gr.Interface(fn=chatbot_response,
                     inputs="text",
                     outputs="text",
                     title="Trained Chatbot",
                     css=css)
iface.launch()
