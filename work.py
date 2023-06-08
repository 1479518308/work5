import torch
from datasets import load_dataset
from transformers import BertModel,BertTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW
from transformers import ErnieForMaskedLM,BertTokenizer
import openai
#加载字典和分词工具

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_dataset(path='lansinuote/ChnSentiCorp', split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']

        return text, label
def collate_fn_class(data):
    token = BertTokenizer.from_pretrained('bert-base-chinese')
    #token = AutoTokenizer.from_pretrained("albert-base-v2")
    #token = BertTokenizer.from_pretrained('nghuyong/ernie-3.0-base-zh')
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    #编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)

    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    #print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids, labels
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids,pretrained):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0])

        out = out.softmax(dim=1)

        return out
def classcify():
    model = Model()
    optimizer = AdamW(model.parameters(), lr=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    dataset = Dataset('train')
    pretrained = BertModel.from_pretrained('bert-base-chinese')
    #pretrained = AutoModelForMaskedLM.from_pretrained("albert-base-v2")
    #pretrained = ErnieForMaskedLM.from_pretrained('nghuyong/ernie-3.0-base-zh')

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=16,
                                         collate_fn=collate_fn_class,
                                         shuffle=True,
                                         drop_last=True)
    model.train()
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader):
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    pretrained = pretrained)

        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 5 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)

            print(i, loss.item(), accuracy)

        if i == 30:
            break

def erenie():
    tokenizer = BertTokenizer.from_pretrained('nghuyong/ernie-3.0-base-zh')
    model = ErnieForMaskedLM.from_pretrained('nghuyong/ernie-3.0-base-zh')
    textlist = ["[MASK][MASK]是中国的首都。",
                "[MASK][MASK][MASK][MASK]是一个成语，最早出自于战国·楚·宋玉《风赋》。 空穴来风（穴：洞）指有孔洞便会进风。 比喻消息和传闻的产生都是有原因和根据的；也比喻消息和传闻毫无根据。",
                "[MASK][MASK][MASK]与红楼梦，三国演义，水浒传并称为中国四大名著。",
                "太阳从[MASK][MASK]升起，西方落下。",
                "[MASK][MASK][MASK][MASK]together with Dream of the Red Mansion, Romance of the Three Kingdoms, and Water Margin, it is known as China's four masterpieces."]

    input_ids = torch.tensor([tokenizer.encode(text=textlist[-1],
                                               add_special_tokens=True)])
    model.eval()
    with torch.no_grad():
        predictions = model(input_ids)[0][0]
    predicted_index = [torch.argmax(predictions[i]).item() for i in range(predictions.shape[0])]
    predicted_token = [tokenizer._convert_id_to_token(predicted_index[i]) for i in
                       range(1, (predictions.shape[0] - 1))]
    #openai(textlist[1])
    print('predict result:\t', predicted_token)

def openai_reply(content, apikey):
    openai.api_key = apikey
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",  # gpt-3.5-turbo-0301
        messages=[
            {"role": "user", "content": content}
        ],
        temperature=0.5,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    # print(response)
    return response.choices[0].message.content

def openai(content):
    ans = openai_reply(content, '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')
    print(ans)

def qa():
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

    #model_name = "deepset/roberta-base-squad2"
    model_name = "Tahasathalia/bigbird_new_finetuning"

    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': 'Why is model conversion important?',
        'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    }
    res = nlp(QA_input)
    print(res)
    qa_pipeline = pipeline(
        "question-answering",
        model="mrm8488/bert-multi-cased-finetuned-xquadv1",
        tokenizer="mrm8488/bert-multi-cased-finetuned-xquadv1"
    )
    print(qa_pipeline(QA_input))
    # b) Load model & tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    qa()