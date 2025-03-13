import csv
import pandas as pd
import numpy as np
import io
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Thiết lập mã hóa đầu ra thành UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Hàm xử lý từng khối dữ liệu và kết nối trực tiếp vào khung dữ liệu tổng
def process_chunks(file_path, chunk_size):
    chunk_iter = pd.read_csv(file_path, encoding='latin1', chunksize=chunk_size, low_memory=False, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    processed_df = pd.DataFrame()

    for i, chunk in enumerate(chunk_iter):
        chunk.columns = [col if col.strip() != '' else f'Unnamed_col_{i}' for i, col in enumerate(chunk.columns)]
        chunk.columns = [f'col_{i}' if 'Unnamed' in col or col.isdigit() else col for i, col in enumerate(chunk.columns)]
        
        # Xử lý giá trị thiếu bằng phương pháp phù hợp
        fill_missing_values(chunk)

        # Chuyển các cột chỉ chứa số sang kiểu số
        for col in chunk.select_dtypes(include='object').columns:
            if chunk[col].str.isnumeric().all():
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        
        chunk.drop_duplicates(inplace=True)
        processed_df = pd.concat([processed_df, chunk], ignore_index=True)
    
    return processed_df

# Hàm điền giá trị thiếu
def fill_missing_values(df):
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            # Kiểm tra nếu cột có giá trị không phải NaN
            if df[col].notna().any():
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)  # Điền giá trị mặc định nếu toàn bộ cột là NaN
        else:
            if df[col].notna().any():
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna('Unknown')  # Điền giá trị mặc định cho cột danh mục

# Xử lý giá trị vô hạn và NaN trong dữ liệu
def clean_infinite_and_nan_values(df):
    # Thay thế các giá trị vô hạn và NaN bằng giá trị hợp lý (ví dụ: trung bình hoặc median)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)  # Điền giá trị thiếu bằng median của cột
        else:
            df[col].fillna('Unknown', inplace=True)  # Điền giá trị thiếu cho cột danh mục

# Sau khi chuẩn hóa dữ liệu, thêm bước kiểm tra giá trị vô hạn và NaN
def handle_infinite_and_nan_in_X(X):
    # Loại bỏ hoặc thay thế các giá trị vô hạn sau khi chuẩn hóa
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)  # Thay thế NaN bằng giá trị mặc định (0)

# Xử lý từng tệp CSV lớn theo khối
chunk_size = 50000
df1 = process_chunks(r'C:\GPT2\DATA\UNSW-NB15_1.csv', chunk_size)
df2 = process_chunks(r'C:\GPT2\DATA\UNSW-NB15_2.csv', chunk_size)
df3 = process_chunks(r'C:\GPT2\DATA\UNSW-NB15_3.csv', chunk_size)
df4 = process_chunks(r'C:\GPT2\DATA\UNSW-NB15_4.csv', chunk_size)

df = pd.concat([df1, df2, df3, df4], ignore_index=True)

if '255' in df.columns:
    df.drop('255', axis=1, inplace=True)

# Loại bỏ giá trị NaN còn lại sau chuyển đổi
df.fillna(0, inplace=True)

print("Thông tin dataframe sau khi ghép:")
print(df.info())

# Chọn cột nhãn
potential_label_columns = ['0.3', '0.13', '0.14', '0.16', '0.17', '0.18', 'col_108', 'col_109']
label_column = None

for col in potential_label_columns:
    if col in df.columns and df[col].nunique() <= 10:
        label_column = col
        break

if label_column:
    print(f"Sử dụng cột '{label_column}' làm nhãn.")
    
    # Đếm số lượng mẫu cho từng lớp
    label_counts = df[label_column].value_counts()

    # Lọc bỏ các lớp có tần suất xuất hiện chỉ bằng 1
    valid_labels = label_counts[label_counts > 1].index
    df = df[df[label_column].isin(valid_labels)]

    print("Thông tin sau khi loại bỏ các lớp có tần suất chỉ bằng 1:")
    print(df[label_column].value_counts())
    
    # **Ánh xạ nhãn về phạm vi liên tục**
    unique_labels = sorted(df[label_column].unique())  # Lấy danh sách các nhãn duy nhất
    label_map = {old: new for new, old in enumerate(unique_labels)}  # Tạo ánh xạ
    print("Ánh xạ nhãn:", label_map)

    # Áp dụng ánh xạ nhãn
    df[label_column] = df[label_column].map(label_map)
    
    attack_types = df[label_column].unique()
    print("Các kiểu tấn công có trong dữ liệu sau ánh xạ nhãn:")
    print(attack_types)

    X = df.drop(label_column, axis=1)
    y = df[label_column]

    # Chỉ thực hiện One-Hot Encoding cho cột có ít giá trị duy nhất
    cols_to_encode = X.select_dtypes(include='category').columns
    cols_to_encode = [col for col in cols_to_encode if X[col].nunique() < 20]

    # One-Hot Encoding dạng sparse
    X = pd.get_dummies(X, columns=cols_to_encode, drop_first=True, sparse=True)

    # Label Encoding cho cột nhãn
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))

    # Chuẩn hóa dữ liệu, chỉ áp dụng cho các cột số
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Sau khi chuẩn hóa, kiểm tra lại dữ liệu
    handle_infinite_and_nan_in_X(X)

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Lưu dữ liệu đã tiền xử lý
    np.savez('Processed_UNSW_NB15_training_set.npz', X=X_train, y=y_train)
    np.savez('Processed_UNSW_NB15_testing_set.npz', X=X_test, y=y_test)

    print("Hoàn thành tiền xử lý và lưu dữ liệu.")
else:
    print("Không tìm thấy cột nhãn hoặc cột tương tự để chia dữ liệu.")
