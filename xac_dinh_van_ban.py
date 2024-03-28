import os
import pickle
import re
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from underthesea import word_tokenize

# Thống kê số lượng data theo nhãn
count = {}
for line in open('data.txt', encoding='utf-8'):
    key = line.split()[0]
    count[key] = count.get(key, 0) + 1
for key in count:
    print(key, count[key])

# Số nhãn tồn tại trong data
print(len(count))

# Thống kê các word xuất hiện ở tất cả các nhãn
total_label = len(count)
vocab = {}
label_vocab = {}
for line in open('data.txt', encoding='utf-8'):
    words = line.split()
    label = words[0]
    if label not in label_vocab:
        label_vocab[label] = {}
    for word in words[1:]:
        label_vocab[label][word] = label_vocab[label].get(word, 0) + 1
        if word not in vocab:
            vocab[word] = set()
        vocab[word].add(label)

count = {}
for word in vocab:
    if len(vocab[word]) == total_label:
        count[word] = min([label_vocab[x][word] for x in label_vocab])

sorted_count = sorted(count, key=count.get, reverse=True)
for word in sorted_count[:100]:
    print(word, count[word])

# Loại bỏ stopwords khỏi dữ liệu + lưu file dùng về sau
stopword = set()
with open('stopwords.txt', 'w', encoding='utf-8') as fp:
    for word in sorted_count[:100]:
        stopword.add(word)
        fp.write(word + '\n')


def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in stopword:
            words.append(word)
    return ' '.join(words)


with open('data.prep', 'w', encoding='utf-8') as fp:
    for line in open('data.txt', encoding='utf-8'):
        line = remove_stopwords(line)
        fp.write(line + '\n')

# Chia tập train/test
test_percent = 0.2
text = []
label = []

for line in open('data.prep', encoding='utf-8'):
    words = line.strip().split()
    label.append(words[0])
    text.append(' '.join(words[1:]))

X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=test_percent, random_state=42)

# Lưu train/test data
with open('train.txt', 'w', encoding='utf-8') as fp:
    for x, y in zip(X_train, y_train):
        fp.write('{} {}\n'.format(y, x))

with open('test.txt', 'w', encoding='utf-8') as fp:
    for x, y in zip(X_test, y_test):
        fp.write('{} {}\n'.format(y, x))

# Encode label
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
print(list(label_encoder.classes_), '\n')
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

# Thêm đường dẫn cho Model
MODEL_PATH = "models"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Thuật toán Naive Bayes
start_time = time.time()
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.8, max_features=None)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
text_clf = text_clf.fit(X_train, y_train)

train_time = time.time() - start_time
print('Done training Naive Bayes in', train_time, 'seconds.')

# Save model
pickle.dump(text_clf, open(os.path.join(MODEL_PATH, "naive_bayes.pkl"), 'wb'))

# Xem kết quả trên từng nhãn
nb_model = pickle.load(open(os.path.join(MODEL_PATH, "naive_bayes.pkl"), 'rb'))
y_pred = nb_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=list(label_encoder.classes_)))

# Định nghĩa các hàm và biến cần thiết cho việc xử lý tiếng Việt

# Định nghĩa từ điển các ký tự có dấu và không dấu
uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"


# Hàm chuyển đổi unicode tổ hợp sang dạng chuẩn
def convert_unicode(txt):
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)


def chuan_hoa_dau_cau_tieng_viet(text):
    text = re.sub(r'([AÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶEÉÈẺẼẸÊẾỀỂỄỆIÍÌỈĨỊOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢUÚÙỦŨỤƯỨỪỬỮỰYÝỲỶỸỴ])([AEIOUY])', r'\1\2',
                  text)

    return text


def text_preprocess(document):
    document = remove_html(document)
    document = convert_unicode(document)
    document = chuan_hoa_dau_cau_tieng_viet(document)
    document = word_tokenize(document, format="text")
    document = document.lower()
    document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ_]', ' ', document)
    document = re.sub(r'\s+', ' ', document).strip()
    return document


def remove_html(txt):
    return re.sub(r'<[^>]*>', '', txt)


# Hàm chính để xử lý nhập liệu và phân loại văn bản
def main():
    # Nhập dữ liệu từ bàn phím hoặc tải lên
    print("---Nhập sự lựa chọn của bạn---")
    print("1: Nhập dữ liệu mà bạn muốn phân loại.")
    print("2: Tải dữ liệu mà bạn muốn phân loại.")

    while True:
        try:
            choice = int(input("Lựa chọn của bạn: "))
            if choice in [1, 2]:
                break
            else:
                print("Vui lòng nhập 1 hoặc 2.")
        except ValueError:
            print("Vui lòng nhập một số nguyên hợp lệ.")

    if choice == 1:
        document = input("Nhập văn bản cần phân loại: ")
    elif choice == 2:
        document = text_upload()
    else:
        print("Lựa chọn không hợp lệ.")
        return

    document_preprocessed = text_preprocess(document)
    document_preprocessed = remove_stopwords(document_preprocessed)
    label = nb_model.predict([document_preprocessed])
    print('Văn bản của bạn thuộc loại:', label_encoder.inverse_transform(label))


if __name__ == "__main__":
    main()
