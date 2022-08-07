import torch 

class QuestionGenerator:
    def __init__(self, model, tokenizer, device, max_input_length, max_output_length):
       
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length 
        
    def text2ids(self, source_text): 
        # prepare data
        source_text = str(source_text)
        source_text = ' '.join(source_text.split())
        # from text to ids
        source = self.tokenizer.batch_encode_plus([source_text], max_length=self.max_input_length, pad_to_max_length=True, 
                                            truncation=True, padding="max_length", return_tensors='pt')
        source_ids = source['input_ids'].squeeze().view(1, -1)
        source_mask = source['attention_mask'].squeeze().view(1, -1)
        
        return {'source_ids': source_ids.to(dtype=torch.long), 'source_mask': source_mask.to(dtype=torch.long)}

    def generate(self, source_text):
        
        data = self.text2ids(source_text)
        
        self.model.eval()
        with torch.no_grad():
            ids = data['source_ids'].to(self.device, dtype = torch.long)
            mask = data['source_mask'].to(self.device, dtype = torch.long)

            generated_ids = self.model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=self.max_output_length, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )

            # get words from ids
            prediction = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            
        return prediction