import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats

# 1. Đọc file CSV chứa dữ liệu thật
df = pd.read_csv("/Users/nguyenvietanh/Desktop/Web/Numpy_project/vietnam_housing_dataset_filtered_hanoi.csv")

# 2. Lọc dữ liệu chỉ cho "Đống Đa"
dc = df[df["Address"].str.contains("Đống Đa", na=False)]

# 3. Xoá những cột không cần thiết và xử lý giá trị không phải số
dc = dc.drop(columns=['Address', 'House direction', 'Balcony direction', 'Legal status', 'Furniture state'])

# 4. Xử lý giá trị thiếu cho các cột số
numeric_cols = dc.select_dtypes(include=[np.number]).columns
dc[numeric_cols] = dc[numeric_cols].fillna(dc[numeric_cols].mean())

# 5. Tách biến độc lập (X) và biến phụ thuộc (y)
X = dc[['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']]
y = dc['Price']

# 6. Tính Z-score và loại bỏ các outliers
z_scores = np.abs(stats.zscore(X))
filtered_entries = (z_scores < 3).all(axis=1)
X_filtered = X[filtered_entries]
y_filtered = y[filtered_entries]

# 7. Chuẩn hóa các biến độc lập
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# 8. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_filtered, test_size=0.2, random_state=42)

# 9. Khởi tạo các mô hình cơ bản (base models)
base_models = [
    ('linear', LinearRegression()),
    ('lasso', Lasso(alpha = 0.1)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42))
]

# 10. Khởi tạo mô hình meta (meta-model) là Linear Regression
meta_model = LinearRegression()

# 11. Khởi tạo và huấn luyện mô hình Stacking
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
stacking_model.fit(X_train, y_train)

# 12. Dự đoán trên tập kiểm tra
y_pred = stacking_model.predict(X_test)

# 13. Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# 14. Dự đoán giá nhà cho ví dụ ngẫu nhiên
Nha_vi_du = pd.DataFrame({
    'Area': [100, 60, 78, 50, 15],
    'Frontage': [12, 15, 20, 8, 5],
    'Access Road': [5, 6, 4, 7, 2],
    'Floors': [2, 4, 3, 5, 1],
    'Bedrooms': [3, 5, 4, 5, 1],
    'Bathrooms': [2, 3, 4, 2, 1]
})

# 15. Chuẩn hóa dữ liệu ví dụ và dự đoán
Nha_vi_du_scaled = scaler.transform(Nha_vi_du)
gia_du_doan = stacking_model.predict(Nha_vi_du_scaled)

# 16. Tạo biểu đồ phân tán giữa giá trị thực tế và giá trị dự đoán
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Giá trị dự đoán')

# 17. Vẽ đường y = x (dự đoán hoàn hảo)
max_val = max(max(y_test), max(y_pred))
min_val = min(min(y_test), min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', label='Dự đoán hoàn hảo (y = x)')

# 18. Thêm tiêu đề và nhãn trục
plt.title(f"Biểu đồ dự đoán giá nhà bằng mô hình Stacking\n$R^2$: {r2:.2f}")
plt.xlabel("Giá trị thực tế (Tỷ VND)")
plt.ylabel("Giá trị dự đoán (Tỷ VND)")

# 19. Thêm chú thích
plt.legend()
plt.grid(True)

# 20. Thêm thông tin MAE và MSE vào biểu đồ
textstr = f'MSE: {mse:.2f}\nMAE: {mae:.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

# 21. Hiển thị biểu đồ
plt.show()

# 22. Kết quả dự đoán
for i, price in enumerate(gia_du_doan):
    print(f"Ngôi nhà thứ {i+1} - Giá dự đoán: {price:.2f} tỷ VND")

# 23. Đánh giá mô hình
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R² (R-squared):", r2)