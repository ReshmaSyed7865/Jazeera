import os
import time
import pandas as pd
from openai import AzureOpenAI
import json
from pathlib import Path

# Azure OpenAI Configuration
deployment_name = 'gpt-4'
api_version = '2024-05-01-preview'
api_base = "https://otherusecases.openai.azure.com/"
api_key = ""

# Initialize Azure OpenAI client
client = AzureOpenAI(api_key=api_key, azure_endpoint=api_base, api_version=api_version)

# Paths
input_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "India Arabic")
output_excel = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jazeera_qa_pairs.xlsx")
progress_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "progress.json")

def load_progress():
    """Load progress from file to resume from last processed file"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"last_processed_file": "", "processed_files": []}

def save_progress(last_file, processed_files):
    """Save progress to file"""
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump({"last_processed_file": last_file, "processed_files": processed_files}, f)

def generate_qa_pairs(text_content, url):
    """Generate 5 descriptive Q&A pairs using Azure OpenAI"""
    prompt = f"""
    أنشئ 5 أزواج من الأسئلة والأجوبة التفصيلية بناءً على المعلومات التالية من موقع طيران الجزيرة. 
    
    المتطلبات:
    1. يجب أن تكون الأسئلة متنوعة وواقعية ومفصلة كما لو كان المسافرون يسألون روبوت الدردشة الخاص بطيران الجزيرة.
    2. يجب أن تكون الأسئلة طويلة بما فيه الكفاية (15-25 كلمة) وتحتوي على تفاصيل محددة.
    3. يجب أن تكون الإجابات شاملة وتفصيلية (50-100 كلمة) وتقدم معلومات كاملة.
    4. قدم الإجابات بلغة عربية سليمة ومهذبة وبأسلوب احترافي.
    5. تأكد من أن الأسئلة والأجوبة مناسبة لاستخدامها في تدريب نموذج ذكاء اصطناعي.
    6. استخدم المعلومات المتاحة في النص لإنشاء أزواج سؤال وجواب دقيقة ومفيدة.
    
    المحتوى:
    {text_content}
    
    URL: {url}
    
    قم بتنسيق الإخراج بالضبط كما يلي (JSON):
    {{"pair_1": {{"question": "السؤال الأول المفصل (15-25 كلمة)", "answer": "الإجابة الأولى المفصلة (50-100 كلمة)"}},
    "pair_2": {{"question": "السؤال الثاني المفصل (15-25 كلمة)", "answer": "الإجابة الثانية المفصلة (50-100 كلمة)"}},
    "pair_3": {{"question": "السؤال الثالث المفصل (15-25 كلمة)", "answer": "الإجابة الثالثة المفصلة (50-100 كلمة)"}},
    "pair_4": {{"question": "السؤال الرابع المفصل (15-25 كلمة)", "answer": "الإجابة الرابعة المفصلة (50-100 كلمة)"}},
    "pair_5": {{"question": "السؤال الخامس المفصل (15-25 كلمة)", "answer": "الإجابة الخامسة المفصلة (50-100 كلمة)"}}}}
    """
    
    messages = [
        {"role": "system", "content": "أنت مساعد متخصص في إنشاء أزواج من الأسئلة والأجوبة التفصيلية لاستخدامها في تدريب نماذج الذكاء الاصطناعي. قم بإنشاء أسئلة مفصلة (15-25 كلمة) وأجوبة شاملة (50-100 كلمة) باللغة العربية الفصحى وبأسلوب احترافي ومهذب. استخدم المعلومات المتاحة لإنشاء محتوى تعليمي ومفيد للمسافرين."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=0.1,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        qa_data = json.loads(result)
        
        # Validate that we have exactly 5 pairs
        if not all(f"pair_{i}" in qa_data for i in range(1, 6)):
            # If missing pairs, create a properly formatted response
            fixed_data = {}
            for i in range(1, 6):
                key = f"pair_{i}"
                if key in qa_data and "question" in qa_data[key] and "answer" in qa_data[key]:
                    fixed_data[key] = qa_data[key]
                else:
                    # Use data from another pair or create placeholder
                    existing_pair = next(iter(qa_data.values())) if qa_data else {"question": "سؤال", "answer": "جواب"}
                    fixed_data[key] = {
                        "question": f"سؤال إضافي {i} حول {url.split('/')[-1]}",
                        "answer": f"إجابة للسؤال {i} بناءً على المعلومات المتاحة."
                    }
            qa_data = fixed_data
            
        return qa_data, url
    except Exception as e:
        print(f"Error generating QA pairs: {e}")
        # Return a default structure in case of error
        default_data = {}
        for i in range(1, 6):
            default_data[f"pair_{i}"] = {
                "question": f"سؤال {i} حول {url.split('/')[-1]}",
                "answer": f"لم نتمكن من إنشاء إجابة بسبب خطأ في المعالجة."
            }
        return default_data, url

def extract_url_from_content(content):
    """Extract URL from file content"""
    for line in content.split('\n'):
        if line.startswith('URL:'):
            return line.replace('URL:', '').strip()
    return "Unknown URL"

def main():
    # Load existing progress
    progress = load_progress()
    last_processed = progress["last_processed_file"]
    processed_files = progress["processed_files"]
    
    # Get all text files
    all_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    total_files = len(all_files)
    
    # Determine starting point
    start_index = 0
    if last_processed and last_processed in all_files:
        start_index = all_files.index(last_processed) + 1
    
    # Load or create DataFrame
    if os.path.exists(output_excel):
        try:
            df = pd.read_excel(output_excel)
            print(f"Loaded existing Excel file with {len(df)} QA pairs")
        except Exception as e:
            print(f"Error loading Excel file: {e}. Creating new file.")
            df = pd.DataFrame(columns=["question", "answer", "url"])
    else:
        df = pd.DataFrame(columns=["question", "answer", "url"])
        print("Created new Excel file for QA pairs")
    
    remaining_files = total_files - start_index
    print(f"Starting from file {start_index + 1} of {total_files} ({remaining_files} files remaining)")
    
    # Process files
    try:
        for i, filename in enumerate(all_files[start_index:], start=start_index):
            if filename in processed_files:
                print(f"Skipping already processed file: {filename}")
                continue
                
            file_path = os.path.join(input_folder, filename)
            
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract URL
                url = extract_url_from_content(content)
                
                print(f"Processing file {i+1}/{total_files}: {filename}")
                print(f"URL: {url}")
                
                # Generate QA pairs
                qa_pairs, url = generate_qa_pairs(content, url)
                
                if qa_pairs:
                    # Add to DataFrame
                    pairs_added = 0
                    for j in range(1, 6):  # 5 QA pairs
                        key = f"pair_{j}"
                        if key in qa_pairs and "question" in qa_pairs[key] and "answer" in qa_pairs[key]:
                            pair = qa_pairs[key]
                            new_row = pd.DataFrame({
                                "question": [pair["question"]],
                                "answer": [pair["answer"]],
                                "url": [url]
                            })
                            df = pd.concat([df, new_row], ignore_index=True)
                            pairs_added += 1
                    
                    # Save progress after each successful file
                    processed_files.append(filename)
                    save_progress(filename, processed_files)
                    
                    # Save DataFrame to Excel
                    df.to_excel(output_excel, index=False)
                    
                    print(f"Added {pairs_added} QA pairs from {filename}")
                    print(f"Progress: {i+1}/{total_files} files processed ({((i+1)/total_files)*100:.1f}%)")
                else:
                    print(f"No QA pairs generated for {filename}")
                
                # Wait 5 seconds between API calls
                print(f"Waiting 5 seconds before next file...")
                time.sleep(5)
                
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                # Save progress even if there's an error
                save_progress(filename, processed_files)
                print("Continuing with next file...")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress saved.")
        print(f"Last processed file: {last_processed}")
        print(f"Resume later to continue from file {start_index + 1}")
    
    print(f"\nProcessing complete or interrupted.")
    print(f"Total QA pairs generated: {len(df)}")
    print(f"Files processed: {len(processed_files)} out of {total_files}")
    print(f"Results saved to: {output_excel}")
    print(f"Progress saved to: {progress_file}")
    
    return df

if __name__ == "__main__":
    main()
