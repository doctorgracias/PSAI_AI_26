import numpy as np
import matplotlib.pyplot as plt


class LinearBinaryClassifier:
    def __init__(self, features_count, lr=0.1):
        self.lr = lr
        self.w = np.random.uniform(-0.5, 0.5, features_count + 1)

    def _add_bias_column(self, X):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        bias = np.ones((X.shape[0], 1))
        return np.concatenate((bias, X), axis=1)

    def _activation(self, z):
        return np.where(z >= 0, 1, 0)

    def predict(self, X):
        Xb = self._add_bias_column(X)
        linear_output = Xb @ self.w
        return self._activation(linear_output)

    def train(self, X, y, epochs=700):
        Xb = self._add_bias_column(X)
        y = np.array(y, dtype=int)
        history = []

        for epoch in range(epochs):
            errors = 0

            for xi, target in zip(Xb, y):
                output = self._activation(np.dot(xi, self.w))
                update = self.lr * (target - output)

                if update != 0:
                    self.w += update * xi
                    errors += 1

            history.append(errors)

            if errors == 0:
                print(f"Обучение завершено на эпохе {epoch+1}")
                break

        return history


# =======================
# ДАННЫЕ (из таблицы)
# =======================

X_data = np.array([
    [3, 4],
    [-3, 4],
    [3, -4],
    [-3, -4]
])

y_data = np.array([1, 1, 0, 0])


# =======================
# ОБУЧЕНИЕ
# =======================

classifier = LinearBinaryClassifier(features_count=2, lr=0.12)
loss_curve = classifier.train(X_data, y_data)


# =======================
# ВИЗУАЛИЗАЦИЯ
# =======================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- График ошибок ---
axes[0].plot(loss_curve, color='darkgreen', linewidth=2)
axes[0].set_title("Динамика количества ошибок")
axes[0].set_xlabel("Эпоха")
axes[0].set_ylabel("Ошибки")
axes[0].grid(alpha=0.3)


# --- Область классификации ---
xmin, xmax = -6, 6
ymin, ymax = -6, 6

grid_x, grid_y = np.meshgrid(
    np.linspace(xmin, xmax, 250),
    np.linspace(ymin, ymax, 250)
)

grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
grid_pred = classifier.predict(grid_points).reshape(grid_x.shape)

axes[1].contourf(grid_x, grid_y, grid_pred,
                 levels=[-0.1, 0.5, 1.1],
                 colors=['#ffcccc', '#cce5ff'],
                 alpha=0.6)

axes[1].contour(grid_x, grid_y, grid_pred,
                levels=[0.5],
                colors='black',
                linestyles='dashed')

colors = ['red' if label == 0 else 'blue' for label in y_data]
axes[1].scatter(X_data[:, 0], X_data[:, 1],
                c=colors,
                edgecolors='black',
                s=120)

# Прямая вручную
if abs(classifier.w[2]) > 1e-6:
    x_line = np.linspace(xmin, xmax, 200)
    y_line = -(classifier.w[0] + classifier.w[1]*x_line) / classifier.w[2]
    axes[1].plot(x_line, y_line, color='black', linewidth=2)

axes[1].set_xlim(xmin, xmax)
axes[1].set_ylim(ymin, ymax)
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")
axes[1].set_title("Линейная граница решения")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =======================
# РЕЖИМ ВВОДА
# =======================

print("\nВведите координаты точки (x1 x2) или 'q' для выхода\n")

while True:
    user_input = input(">> ").strip()

    if user_input.lower() in ('q', 'exit'):
        break

    try:
        x1, x2 = map(float, user_input.split())
        prediction = classifier.predict([x1, x2])[0]
        print(f"Класс точки: {prediction} (0 — нижняя область, 1 — верхняя)")
    except:
        print("Ошибка ввода. Введите два числа через пробел.")
