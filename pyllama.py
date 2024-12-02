import os
from datetime import datetime

import ollama
import requests
from tabulate import tabulate
import subprocess

# Configuration
OLLAMA_SERVER_URL = "http://localhost:11434"  # Replace with your actual server URL

def get_installed_models():
    response = requests.get(f"{OLLAMA_SERVER_URL}/api/tags/")
    if response.status_code == 200:
        models = response.json().get("models", [])
        return models
    else:
        print(f"Failed to retrieve models: {response.status_code}")
        return []


def bytes_to_gb(byte_size):
    return byte_size / (1024 ** 3)


def format_model_details(model):
    details = model.get("details", {})
    family = details.get("family", "")
    param_size = details.get("parameter_size", "")
    quant_level = details.get("quantization_level", "")
    # parent_model = details.get("parent_model", "")
    size = round(model.get("size", "")/1024/1024/1024, 2)
    try:
        param_size_gb = bytes_to_gb(int(param_size))
        param_size_display = f"{param_size_gb:.2f} GB"
    except (ValueError, TypeError):
        param_size_display = param_size

    return {
        "Name": model.get("name", ""),
        "Family": family,
        "Parameter Size": param_size_display,
        "Quantization Level": quant_level,
        "Size": f"{size} GB",
    }


def print_model_menu(models):
    headers = ["Index", "Name", "Family", "Param. Size", "Quant. Lvl", "Size"]
    table_data = []

    for index, model in enumerate(models, start=1):
        formatted_details = format_model_details(model)
        table_data.append([index] + list(formatted_details.values()))

    print("\nInstalled Models:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def select_model(models):
    while True:
        try:
            selection = int(input("Select a model by number: "))
            if 1 <= selection <= len(models):
                return models[selection - 1]
            else:
                print("Invalid selection. Please choose a valid number.")
        except ValueError:
            print("Please enter a valid number.")


def get_user_parameters():
    temperature = float(input("Enter temperature (e.g., 0.7): "))
    num_ctx = int(input("Enter num_ctx (e.g., 2048): "))
    top_k = int(input("Enter top_k (e.g., 50): "))
    top_p = float(input("Enter top_p (e.g., 0.9): "))
    min_p = float(input("Enter min_p (e.g., 0.01): "))

    return {
        "temperature": temperature,
        "num_ctx": num_ctx,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p
    }


def write_to_model_file(model_name, original_model_name, parameters):
    with open(f"{model_name}", "w") as file:
        file.write(f"FROM {original_model_name}\n")
        for key, value in parameters.items():
            file.write(f"PARAMETER {key} {value}\n")


def create_ollama_model_with_config(model_name, config_file):
    try:
        subprocess.run(["ollama", "create", model_name, "-f", config_file], check=True)
        print(f"Model {model_name} created successfully.")
    except Exception as e:
        print(f"Error creating model: {e}")


def secondary_menu(selected_model):
    while True:
        print("\nSecondary Menu:")
        print("1. Run Model")
        print("2. Show Model Details")
        print("3. Copy Model")
        print("4. Back to Main Menu")
        print("5. Exit")
        choice = input("Enter your choice (1/2/3/4): ")

        if choice == '1':
            run_model(selected_model["name"])
        elif choice == '2':
            show_model_details(selected_model["name"])
        elif choice == '3':
            new_model_name = input("Enter the name for the new model: ")
            parameters = get_user_parameters()
            config_file = f"{new_model_name}.modelfile"
            write_to_model_file(config_file, selected_model["name"], parameters)
            create_ollama_model_with_config(new_model_name, config_file)
        elif choice == '4':
            break
        elif choice == '5':
            return -1
        else:
            print("Invalid option. Please try again.")


def run_model(model_name):
    print(f"Running model: {model_name}")
    user_prompt = ""

    # Create a log file with current date and time
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"Session_{current_time}.log")

    try:
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            while user_prompt.lower() != "exit":
                user_prompt = input("\n\n\nEnter a prompt > ")

                # Log the user prompt
                log_file.write(f"User: {user_prompt}\n")
                log_file.flush()  # Ensure the prompt is written immediately

                # Log the model's response
                log_file.write("Assistant: ")
                for chunk in ollama.chat(model=model_name, messages=[{"role": "user", "content": user_prompt}],
                                         stream=True):
                    content = chunk["message"]["content"]
                    print(content, end="", flush=True)
                    log_file.write(content)
                    log_file.flush()  # Continuously save the response

                log_file.write("\n--------------------------------------------------------------------------------\n")
                log_file.flush()

                print("\n--------------------------------------------------------------------------------")
    except Exception as e:
        print(f"An error occurred: {e}")
        # The log file will be saved up to the point of the error due to continuous flushing
    finally:
        print(f"Session log saved to: {log_file_path}")


def show_model_details(model_name):
    print(f"Details for model: {model_name}")
    models = get_installed_models()
    for model in models:
        if model["name"] == model_name:
            print("Model Name:", model["name"])
            print("Family:", model.get("details", {}).get("family", ""))
            print("Parameter Size:", model.get("details", {}).get("parameter_size", ""))
            print("Quantization Level:", model.get("details", {}).get("quantization_level", ""))
            print("Size:", bytes_to_gb(model["size"]))



def main():
    check = 0
    while check!= -1:
        models = get_installed_models()
        if not models:
            print("No models found.")
            return

        print_model_menu(models)
        selected_model = select_model(models)
        check = secondary_menu(selected_model)


if __name__ == "__main__":
    main()