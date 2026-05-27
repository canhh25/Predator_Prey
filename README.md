# Predator-Prey Gridworld: Multi-Agent System Environment

Dự án mô phỏng bài toán rượt đuổi (Predator-Prey) trong không gian dạng lưới (Gridworld), được xây dựng dựa trên chuẩn API `ParallelEnv` của PettingZoo. 

Môi trường được thiết kế như một nền tảng cơ sở (baseline) để thử nghiệm, đánh giá và huấn luyện các thuật toán Học tăng cường đa thể (Multi-Agent Reinforcement Learning - MARL) và các mô hình LLM-as-Agent.

## Cấu trúc thư mục

Kiến trúc dự án được phân tách rõ ràng giữa Môi trường (Game Engine) và Thuật toán (Agents), đảm bảo tính module hóa cao:

```text
Predator_Prey/
│
├── env/
│   └── gridworld.py       # Chứa class GridWorldEnv quản lý tọa độ, không gian và luật lệ.
│
├── agents/
│   ├── predator_algo.py   # Bộ não của các Predators (Cá mập).
│   └── prey_algo.py       # Bộ não của Prey (Con mồi).
│
├── main.py                # File thực thi vòng lặp game, kết nối Agents vào Environment.
└── README.md
