# Predator-Prey Gridworld: Multi-Agent System Environment

Dự án mô phỏng bài Predator-Prey trong không gian dạng lưới, được xây dựng dựa trên chuẩn API `ParallelEnv` của PettingZoo. 

Môi trường được thiết kế như một nền tảng cơ sở để thử nghiệm, đánh giá và huấn luyện các thuật toán Multi-Agent Reinforcement Learning (MARL).

## Cấu trúc thư mục

Kiến trúc dự án:

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
