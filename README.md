# Research-Paper-Extraction

Extract key information from research papers using free LLM models.

---

## Setup (Python runtime)

1. Download **Miniconda** from [here](https://www.anaconda.com/download/success).  
2. Follow the installation steps. You don’t need to change any default settings.  

   ![Miniconda Installation](https://github.com/user-attachments/assets/c4fb5535-0965-4b40-9d53-b555d0f8cfe9)

3. Once installed, search for **Anaconda Prompt** in the Windows search bar.  

   ![Anaconda Prompt](https://github.com/user-attachments/assets/09640066-cbc6-4a8c-9428-b455ddeceac0)

4. Open the **Anaconda Prompt** and run the following commands to set up the environment:

   ```bash
   conda create -n llm_extract_paper python=3.11
   conda activate llm_extract_paper
   pip install -r <your_path>/Research-Paper-Extraction/requirements.txt

## Setup (LLM API Key)

1. Go to [groq](https://console.groq.com/keys) to generate your API key. Sign in using your google account.
   
   ![Miniconda Installation](https://github.com/user-attachments/assets/36c82949-c947-4bf2-b9aa-427f2e7de74d)
   
2. Copy the API key and paste into <your_path/Research-Paper-Extraction/env.

## Steps-to-Use

1. Open **Anaconda Prompt** and activate the environment:

   ```bash
   conda activate llm_extract_paper

2. Place all research papers you want to process inside the media folder.
3. (Optional) Update prompt_read and prompt_synthesis if you want to customize the features to be extracted.
4. Run the main script:

   ```bash
   python <your_path>/Research-Paper-Extraction/main.py

6. The extracted results will be saved in <your_path>/Research-Paper-Extraction/output.

## Results (Sample)
![Result](https://github.com/user-attachments/assets/86769b90-a98e-4a04-91d5-eabab0cd3501)
