# Результаты дневного M5

Финальный private WRMSSE поздней отправки — **0.58009**. Локальный public WRMSSE —
**0.613284**, историческое окно подбора — **0.489545**.

| Папка | Содержимое |
|---|---|
| [final_m5_frozen_ensemble_v1](final_m5_frozen_ensemble_v1) | Финальный рецепт, 12 fit-протоколов, метрики и манифест submission |
| [ensemble_horizon_week1_3200_v1](ensemble_horizon_week1_3200_v1) | Веса по неделям и калибровка 1.006 |
| [lgbm_horizon_week1_trial9_52x28_3200_v1](lgbm_horizon_week1_trial9_52x28_3200_v1) | Дополнительный запуск первой недели: 2076 деревьев |
| [lgbm_horizon_weeks_trial9_52x28_v1](lgbm_horizon_weeks_trial9_52x28_v1) | Первые четыре недельных специалиста |
| [optuna_lgbm_67_cpu_52x28_v1](optuna_lgbm_67_cpu_52x28_v1) | Поиск на плотности 28: 13 завершённых попыток |
| [lgbm_finalists_density_v1](lgbm_finalists_density_v1) | Плотности 16/28, перенос финалистов на другие окна и первые ансамбли моделей |
| [optuna_lgbm_67_cpu_52x8_v1](optuna_lgbm_67_cpu_52x8_v1) | Первый подбор LightGBM: 29 завершённых попыток |
| [kaggle_session_2026-10-25](kaggle_session_2026-10-25) | Цена целевой недели и отклонённые EWMA/короткие лаги |
| [kaggle_migration_v9](kaggle_migration_v9) | Проверка расхождения признаков при переносе CatBoost |

Остальные папки содержат промежуточные версии CatBoost и ранние LightGBM. Их значения
относятся к указанным в протоколах датам и настройкам; это не оценки финального ансамбля моделей.

В Git сохраняются небольшие таблицы, планы, параметры и графики. Полные прогнозы,
модели и submission доступны в [Release v1.0.0](https://github.com/dannya220/demand-planner/releases/tag/v1.0.0).
Список и контрольные суммы архивов — в [release_assets.json](../release_assets.json).
