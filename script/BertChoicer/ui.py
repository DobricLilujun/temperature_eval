import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
)


# 模拟从后台分类模型获取标签比重和对应温度
def get_classification_probabilities(prompt):
    # 假设这里是从分类模型获取的标签比重
    labels = ["Label 1", "Label 2", "Label 3"]
    probabilities = np.random.dirichlet(
        np.ones(len(labels)), size=1
    ).flatten()  # 模拟概率分布
    return dict(zip(labels, probabilities))


def get_optimal_temperature(probabilities):
    # 假设根据概率计算温度的简单逻辑
    temperature = np.sum(probabilities) / len(probabilities)
    return round(temperature, 2)


def get_language_model(probabilities):
    # 根据概率的最大值选择语言模型
    max_label = max(probabilities, key=probabilities.get)
    if max_label == "Label 1":
        return "GPT-3"
    elif max_label == "Label 2":
        return "GPT-4"
    else:
        return "LLaMA"


class UserInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prompt Analyzer")

        # 创建布局
        self.layout = QVBoxLayout()

        # 输入提示框
        self.prompt_label = QLabel("Enter Prompt:")
        self.layout.addWidget(self.prompt_label)

        self.prompt_input = QLineEdit(self)
        self.layout.addWidget(self.prompt_input)

        # 按钮
        self.analyze_button = QPushButton("Analyze", self)
        self.analyze_button.clicked.connect(self.analyze_prompt)
        self.layout.addWidget(self.analyze_button)

        # 输出显示
        self.result_label = QLabel("Result:")
        self.layout.addWidget(self.result_label)

        self.result_output = QTextEdit(self)
        self.result_output.setReadOnly(True)
        self.layout.addWidget(self.result_output)

        # 设置窗口布局
        self.setLayout(self.layout)

    def analyze_prompt(self):
        prompt = self.prompt_input.text()

        if prompt:
            # 获取分类概率分布
            probabilities = get_classification_probabilities(prompt)

            # 计算最优温度
            optimal_temperature = get_optimal_temperature(list(probabilities.values()))

            # 选择语言模型
            model = get_language_model(probabilities)

            # 显示结果
            result_text = f"Label Probabilities: {probabilities}\n"
            result_text += f"Optimal Temperature: {optimal_temperature}\n"
            result_text += f"Chosen Language Model: {model}"
            self.result_output.setText(result_text)
        else:
            self.result_output.setText("Please enter a valid prompt.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UserInterface()
    window.show()
    sys.exit(app.exec())  # Notice the change here: exec_() -> exec()
