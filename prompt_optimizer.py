def improve_prompt_i2v(prompt, target, model, tokenizer):
    # Strict English instruction prompt
    system_prompt = f"""
    You are a professional prompt optimizer. Your task is:
    
    1. Rewrite the user-provided prompt to start with {target}, focus on the detailed description of the main characters, including facial details, clothing, actions, etc., weaken the background description, but do not omit information.
    
    2. Ensure the logic of the expression, you can add some subtle reasonable facial changes.
    
    3. Only output the improved instruction - no explanations, notes, or additional content.
    """

    full_prompt = f"{system_prompt}\n\nOriginal prompt: {prompt}"
    
    messages = [{"role": "user", "content": full_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # Disable thinking mode prefix
        enable_thinking=False         # Disable thinking process
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate with strict constraints
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1000,       # Limit response length
        temperature=0.1,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Clean decoding
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    # 后处理：确保输出以 {target} 开头
    if not content.startswith(target):
        target_index = content.find(target)
        if target_index != -1:
            content = content[target_index:]

    return content.strip()

def improve_prompt_t2i(prompt, target, model, tokenizer):
    # Strict English instruction prompt
    system_prompt = f"""
    You are a professional prompt optimizer. Given a video description instruction, you need to extract key elements from it, 
    including human faces, clothing, environments, etc., and arrange them in the form of phrases.
    
    For example, a reference output example: Real photography, a girl, RAW photo, Korean portrait photography, korean style, close-up.

    Ignore descriptions of feet, shoes, and background characters, and add a 'close-up' at the end of your output instead.

    Only output the improved instruction - no explanations, notes, or additional content!
    """

    full_prompt = f"{system_prompt} \n\nThe video prompt is: {prompt}"
    
    messages = [{"role": "user", "content": full_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # Disable thinking mode prefix
        enable_thinking=False         # Disable thinking process
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate with strict constraints
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1000,       # Limit response length
        temperature=0.1,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Clean decoding
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    if not content.startswith(target):
        target_index = content.find(target)
        if target_index != -1:
            content = content[target_index:]
    
    if "close-up" not in content:
        content += ", close-up"

    return "fcsks fxhks fhyks, " + content.strip().replace("</think>\n\n", "")