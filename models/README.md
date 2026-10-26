# Сохранённые модели

Скачать [Release v1.0.0](https://github.com/dannya220/demand-planner/releases/tag/v1.0.0):

- `demand-planner_models_v1.zip` — только модели и данные их запуска;
- `demand-planner_final_checkpoint.zip` — полный запуск с прогнозами и submission.

В архиве 12 моделей LightGBM: шесть public и шесть private. Распаковать в корень проекта,
сохраняя пути `models/final_m5_frozen_ensemble_v1/{public,private}/`.

```bash
python scripts/verify_models.py .
```

Проверка сверяет SHA-256, размер, число деревьев и список признаков с
[manifest.json](manifest.json). Она не загружает LightGBM и не строит прогнозы.

Параметры, планы обучения и рецепт смеси находятся в
[результатах финального запуска](../results/m5_wrmsse/final_m5_frozen_ensemble_v1).
Для продолжения notebook 10 подойдёт полный checkpoint из Release. При использовании
архива только моделей его распакованную папку можно указать в `RESUME_FROM`: сохранённые
fit-протоколы позволят построить прогнозы без нового обучения.
