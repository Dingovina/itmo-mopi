# Лабораторная 4 — трассировка путей (C)

Минимальный path tracer: Lambert + зеркало, area lights, антиалиасинг, PPM.

## Структура проекта

```
lab_4/
├── Makefile              # сборка
├── src/                  # исходный код
│   └── main.c
├── build/                # бинарник (генерируется, в .gitignore)
│   └── pathtrace
├── assets/scenes/        # описание сцен
│   └── scene.json
├── renders/              # выходные PPM
└── docs/                 # задание
    ├── task.md
    └── task_image.png
```

Команды запускаются из корня `lab_4/` — пути к сцене и выходу задаются относительно него.

## Сборка и запуск

```bash
make              # build/pathtrace
make test         # быстрый тест 64×64
make run          # рендер по параметрам scene.json → renders/output.ppm
```

Полный рендер:

```bash
./build/pathtrace --spp 100 --max-depth 8 --out renders/output.ppm
```

Параметры CLI: `--scene`, `--out`, `--w`, `--h`, `--spp`, `--max-depth`, `--seed`.

Сцена редактируется в `assets/scenes/scene.json` (материалы, треугольники, камера, render).

Подробное описание алгоритма — в комментариях `src/main.c` и в `docs/task.md`.
