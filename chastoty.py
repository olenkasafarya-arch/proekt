import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class SignalProcessor:
    def __init__(self):
        """
        1. ИСХОДНЫЕ ДАННЫЕ
        """
        # Число анализируемых и включенных в базу сигналов
        self.N_sig = 4

        # Количество отсчетов сигнала на интервале
        self.K_int = 101
        self.ik = np.arange(self.K_int)  # 0..K_int-1

        # Число формируемых копий модельного сигнала на интервале
        self.L = 10
        self.il = np.arange(self.L)  # 0..L-1

        """
        2.1 Параметры реального (наблюдаемого) сигнала
        """
        # Действительные значения относительной нестабильности частоты сигнала
        self.sigma_0_otn = 2e-3

        # Действительные значения средней частоты сигнала
        self.omega_0_0 = 1000

        # Действительные значения СКО случайного процесса
        self.sigma_0_0 = self.sigma_0_otn * self.omega_0_0

        """
        2.2 Параметры для формирования реального случайного процесса
        """
        # Закон распределения в случайном процессе (0-нормальное, 1-логистическое, 2-Лапласа, 3-гамма)
        self.in0 = 0

        # Параметры распределений
        self.mu_s = 0
        self.mu_00 = 0

        # Нормальное распределение (нрм)
        self.sigma_irm = self.sigma_0_0

        # Логистическое распределение (лгс)
        self.m_lgs = 0
        self.s_lgs = (np.sqrt(3) * self.sigma_0_0) / np.pi

        # Распределение Лапласа (лпл)
        self.m_lpl = 0
        self.s_lpl = self.sigma_0_0 / np.sqrt(2)

        # Гамма-распределение (гам)
        self.m_gam = 0
        self.s_gam = self.sigma_0_0 ** 2

    def print_initial_parameters(self):
        """Вывод всех исходных параметров"""
        print("=" * 60)
        print("1. ИСХОДНЫЕ ДАННЫЕ")
        print("=" * 60)
        print(f"Число анализируемых сигналов (N_сиг): {self.N_sig}")
        print(f"Количество отсчетов на интервале (K_инт): {self.K_int}")
        print(f"Число копий модельного сигнала (L): {self.L}")

        print("\n" + "=" * 60)
        print("2.1 ПАРАМЕТРЫ РЕАЛЬНОГО СИГНАЛА")
        print("=" * 60)
        print(f"Относительная нестабильность частоты (σ_0_отн): {self.sigma_0_otn}")
        print(f"Средняя частота сигнала (ω_0): {self.omega_0_0}")
        print(f"СКО случайного процесса (σ_0): {self.sigma_0_0}")

        print("\n" + "=" * 60)
        print("2.2 ПАРАМЕТРЫ РАСПРЕДЕЛЕНИЙ")
        print("=" * 60)
        print(f"Тип распределения (in0): {self.in0} - {self.get_distribution_name(self.in0)}")
        print(f"Математическое ожидание (μ_s): {self.mu_s}")

    def get_distribution_name(self, in_val):
        """Возвращает название распределения по коду"""
        names = {
            0: "НОРМАЛЬНОЕ (нрм)",
            1: "ЛОГИСТИЧЕСКОЕ (лгс)",
            2: "ЛАПЛАСА (лпл)",
            3: "ГАММА (гам)"
        }
        return names.get(in_val, "НЕИЗВЕСТНО")

    """
    2.2 Функции для формирования последовательности измеренных значений частоты
    """

    def omega_rnm(self, K_cit, in0):
        """
        ω_РНМ - равномерное распределение
        """
        return np.random.uniform(0, 1, K_cit)

    def omega_izm_lpc(self, s_lpc, omega_rnm_vals):
        """
        ω_ИЗМ,ЛПЦ - распределение Лапласа через преобразование
        """
        result = np.zeros_like(omega_rnm_vals)
        for i, omega_val in enumerate(omega_rnm_vals):
            if omega_val <= 0.5:
                result[i] = s_lpc * np.log(2 * omega_val)
            else:
                result[i] = -s_lpc * np.log(2 - 2 * omega_val)
        return result

    def mom(self, K, zero, sigma):
        """Нормальное распределение (нрм)"""
        return np.random.normal(zero, sigma, K)

    def nogis(self, K, zero, s_lgc):
        """Логистическое распределение (лгс)"""
        return stats.logistic.rvs(zero, s_lgc, size=K)

    def rganma(self, K, s_gam):
        """Гамма-распределение (гам)"""
        return stats.gamma.rvs(2, scale=np.sqrt(s_gam), size=K)

    def omega_izm_isk(self, in_val):
        """
        Основная функция формирования последовательности измеренных значений частоты
        """
        if in_val == 0:
            # Нормальное распределение
            omega = self.mom(self.K_int, 0, self.sigma_irm)
        elif in_val == 1:
            # Логистическое распределение
            omega = self.nogis(self.K_int, 0, self.s_lgs)
        elif in_val == 2:
            # Распределение Лапласа
            omega_rnm = self.omega_rnm(self.K_int, in_val)
            omega = self.omega_izm_lpc(self.s_lpl, omega_rnm)
        elif in_val == 3:
            # Гамма-распределение
            omega = self.rganma(self.K_int, self.s_gam)
        else:
            raise ValueError(f"Неверное значение in_val: {in_val}")

        # Добавляем среднюю частоту (ω_ik ← ω_ik + ω_0)
        omega += self.omega_0_0

        return omega

    def plot_omega_izm_isk(self):
        """
        Построение графика ω_изм.исх_ik - наблюдаемого сигнала
        """
        # Создаем наблюдаемый сигнал
        omega_izm_isk = self.omega_izm_isk(self.in0)

        # Вычисляем статистики
        M_omega = np.max(omega_izm_isk)
        m_omega = np.min(omega_izm_isk)
        mean_omega = np.mean(omega_izm_isk)
        std_omega = np.std(omega_izm_isk, ddof=1)

        print("\n" + "=" * 60)
        print("НАБЛЮДАЕМЫЙ СИГНАЛ: ω_изм.исх_ik")
        print("=" * 60)
        print(f"Тип распределения: {self.get_distribution_name(self.in0)}")
        print(f"Размер сигнала: {len(omega_izm_isk)} отсчетов")
        print(f"Максимальная частота (M_ω): {M_omega:.2f}")
        print(f"Минимальная частота (m_ω): {m_omega:.2f}")
        print(f"Средняя частота: {mean_omega:.2f}")
        print(f"СКО: {std_omega:.2f}")

        # Строим график
        plt.figure(figsize=(12, 6))
        plt.plot(self.ik, omega_izm_isk, 'b-', linewidth=2, label='ω_изм.исх_ik')

        # Добавляем среднюю линию и зону ±σ
        plt.axhline(y=self.omega_0_0, color='r', linestyle='--',
                    linewidth=2, label=f'ω₀ = {self.omega_0_0}')
        plt.axhline(y=mean_omega, color='g', linestyle=':',
                    linewidth=2, label=f'Выборочное среднее = {mean_omega:.1f}')

        plt.fill_between(self.ik, self.omega_0_0 - self.sigma_0_0,
                         self.omega_0_0 + self.sigma_0_0, alpha=0.2,
                         color='red', label=f'±σ₀ = ±{self.sigma_0_0:.1f}')

        plt.xlabel('ik', fontsize=12)
        plt.ylabel('ω_изм.исх_ik', fontsize=12)
        plt.title(f'Наблюдаемый сигнал: ω_изм.исх_ik ({self.get_distribution_name(self.in0)})',
                  fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Устанавливаем пределы для лучшего отображения
        y_margin = 3 * self.sigma_0_0
        plt.ylim(self.omega_0_0 - y_margin, self.omega_0_0 + y_margin)

        plt.tight_layout()
        plt.show()

        return omega_izm_isk

    def plot_omega_izm_uch(self, omega_izm_isk):
        """
        2.4 Построение графика ω_изм.уч - упорядоченной последовательности
        """
        # Сортируем сигнал: ω_изм.уч = sort(ω_изм.исх)
        omega_izm_uch = np.sort(omega_izm_isk)

        # Вычисляем статистики для упорядоченного сигнала
        M_omega_uch = np.max(omega_izm_uch)
        m_omega_uch = np.min(omega_izm_uch)
        mean_omega_uch = np.mean(omega_izm_uch)

        print("\n" + "=" * 60)
        print("УПОРЯДОЧЕННЫЙ СИГНАЛ: ω_изм.уч")
        print("=" * 60)
        print(f"Максимальная частота (M_ω): {M_omega_uch:.2f}")
        print(f"Минимальная частота (m_ω): {m_omega_uch:.2f}")
        print(f"Средняя частота: {mean_omega_uch:.2f}")

        # Строим график как на картинке
        plt.figure(figsize=(12, 6))

        # Основной график упорядоченной последовательности
        plt.plot(self.ik, omega_izm_uch, 'b-', linewidth=2, label='ω_изм.уч')

        # Настройки как на картинке
        plt.xlabel('ik', fontsize=12)
        plt.ylabel('ω_изм.уч, ik, 0', fontsize=12)
        plt.title('ω_изм.уч, ik, 0', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Устанавливаем пределы как на картинке
        plt.ylim(990, 1010)
        plt.xlim(0, 100)

        # Устанавливаем метки на оси Y как на картинке
        y_ticks = [990, 995, 1000, 1005, 1010]
        y_labels = ['990', '995', '1×10$^3$', '1.005×10$^3$', '1.01×10$^3$']
        plt.yticks(y_ticks, y_labels)

        # Добавляем сетку для лучшей читаемости
        plt.grid(True, alpha=0.3)

        # Размещаем подпись как на картинке
        plt.text(102, 1000, 'ω_изм.уч, ik, 0', fontsize=10, va='center')

        plt.tight_layout()
        plt.show()

        return omega_izm_uch

    """
    3. ФОРМИРОВАНИЕ МОДЕЛЬНЫХ СИГНАЛОВ
    """

    def calculate_sample_statistics(self, omega_izm_isk):
        """
        3.1 Определение среднего и дисперсии выборки наблюдаемого сигнала
        """
        # Выборочное среднее: Mω_изм.выб = mean(ω_изм.исх)
        M_omega_izm_vyb = np.mean(omega_izm_isk)

        # Выборочное СКО: σ_изм.выб = stdev(ω_изм.исх)
        sigma_izm_vyb = np.std(omega_izm_isk, ddof=1)

        # Выборочная дисперсия
        variance_izm_vyb = np.var(omega_izm_isk, ddof=1)

        print("\n" + "=" * 60)
        print("3.1 СТАТИСТИКИ ВЫБОРКИ НАБЛЮДАЕМОГО СИГНАЛА")
        print("=" * 60)
        print(f"Выборочное среднее (Mω_изм.выб): {M_omega_izm_vyb:.8f} × 10³")
        print(f"Выборочное СКО (σ_изм.выб): {sigma_izm_vyb:.8f}")
        print(f"Выборочная дисперсия: {variance_izm_vyb:.8f}")
        print(f"Действительное среднее (ω₀): {self.omega_0_0}")
        print(f"Действительное СКО (σ₀): {self.sigma_0_0}")

        # Сравнение с действительными значениями
        mean_error = abs(M_omega_izm_vyb - self.omega_0_0)
        std_error = abs(sigma_izm_vyb - self.sigma_0_0)

        print(f"\nОшибка среднего: {mean_error:.8f}")
        print(f"Ошибка СКО: {std_error:.8f}")

        return M_omega_izm_vyb, sigma_izm_vyb, variance_izm_vyb

    def generate_model_signals(self, sigma_izm_vyb):
        """
        3.2 Формирование исходных последовательностей значений частоты модельных сигналов
        """
        print("\n" + "=" * 60)
        print("3.2 ФОРМИРОВАНИЕ МОДЕЛЬНЫХ СИГНАЛОВ")
        print("=" * 60)

        # Создаем матрицу для хранения всех модельных сигналов
        omega_mod_isk = np.zeros((self.N_sig, self.K_int * self.L))

        # Параметры для модельных распределений
        sigma_nrm_mod = sigma_izm_vyb  # Нормальное
        s_lgs_mod = (np.sqrt(3) * sigma_izm_vyb) / np.pi  # Логистическое
        s_lpl_mod = sigma_izm_vyb / np.sqrt(2)  # Лапласа
        s_gam_mod = sigma_izm_vyb ** 2  # Гамма

        # Генерируем сигналы для каждого типа распределения
        for in_val in range(self.N_sig):
            if in_val == 0:
                # Нормальное распределение
                signal = self.mom(self.K_int * self.L, 0, sigma_nrm_mod)
            elif in_val == 1:
                # Логистическое распределение
                signal = self.nogis(self.K_int * self.L, 0, s_lgs_mod)
            elif in_val == 2:
                # Распределение Лапласа
                omega_rnm = self.omega_rnm(self.K_int * self.L, in_val)
                signal = self.omega_izm_lpc(s_lpl_mod, omega_rnm)
            elif in_val == 3:
                # Гамма-распределение
                signal = self.rganma(self.K_int * self.L, s_gam_mod)

            omega_mod_isk[in_val] = signal

        return omega_mod_isk

    def plot_model_signals_comparison(self, omega_mod_isk):
        """
        Визуализация модельных сигналов
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        distribution_names = ['Нормальное', 'Логистическое', 'Лапласа', 'Гамма']
        colors = ['blue', 'red', 'green', 'purple']

        # Показываем только первые K_int отсчетов для наглядности
        for i in range(self.N_sig):
            signal = omega_mod_isk[i, :self.K_int]
            axes[i].plot(self.ik, signal, color=colors[i], linewidth=2)
            axes[i].set_title(f'Модельный сигнал: {distribution_names[i]}')
            axes[i].set_xlabel('Отсчеты')
            axes[i].set_ylabel('ω')
            axes[i].grid(True, alpha=0.3)

            # Добавляем статистики на график
            stats_text = f"Mean: {np.mean(signal):.3f}\nStd: {np.std(signal, ddof=1):.3f}"
            axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

    def calculate_model_statistics(self, model_signals):
        """
        Вычисление статистик модельных сигналов согласно скриншотам
        """
        print("\n" + "=" * 60)
        print("СТАТИСТИКИ МОДЕЛЬНЫХ СИГНАЛОВ")
        print("=" * 60)

        # Статистики для каждого типа распределения
        distributions = [
            (0, "нрм", "ω_мод.нрм.иск"),  # нормальное
            (1, "лгс", "ω_мод.пгс.нсх"),  # логистическое
            (2, "лпл", "мод.пш.иск"),  # лапласа
            (3, "гам", "ω_мод.гам.нсх")  # гамма
        ]

        results = {}

        for i, (dist_code, dist_name, var_name) in enumerate(distributions):
            signal = model_signals[dist_code]
            mean_val = np.mean(signal)
            std_val = np.std(signal, ddof=1)

            results[dist_name] = (mean_val, std_val)

            print(f"\n({i + 1})" if i > 0 else "")

            if dist_name == "нрм":
                print(f"{var_name} := ω_мод.иск")
                print(f"М_ω_мод.{dist_name} := mean({var_name})")
                print(f"М_ω_мод.{dist_name} = {mean_val:.8f}")
                print(f"σ_мод.{dist_name} := stdev({var_name})")
                print(f"σ_мод.{dist_name} = {std_val:.8f}")

            elif dist_name == "лгс":
                print(f"{var_name} := ω_мод.нсх")
                print(f"М_ω_мод.{dist_name} := mean({var_name})")
                print(f"М_ω_мод.{dist_name} = {mean_val:.8f}")
                print(f"σ_мод.{dist_name} := stdev({var_name})")
                print(f"σ_мод.{dist_name} = {std_val:.8f}")

            elif dist_name == "лпл":
                print(f"{var_name} := мод.иск")
                print(f"М_ω_мод.{dist_name} := mean({var_name})")
                print(f"М_ω_мод.{dist_name} = {mean_val:.8f}")
                print(f"σ_мод.{dist_name} := stdev({var_name})")
                print(f"σ_мод.{dist_name} = {std_val:.8f}")

            elif dist_name == "гам":
                print(f"{var_name} := ω_мод.нсх")
                print(f"М_ω_мод.{dist_name} := mean({var_name})")
                print(f"М_ω_мод.{dist_name} = {mean_val:.8f}")
                print(f"σ_мод.{dist_name} := stdev({var_name})")
                print(f"σ_мод.{dist_name} = {std_val:.8f}")

        return results

    def normalize_model_signals(self, model_signals, M_omega_vyb, sigma_vyb):
        """
        Нормализация модельных сигналов по формулам
        """
        print("\n" + "=" * 60)
        print("НОРМАЛИЗАЦИЯ МОДЕЛЬНЫХ СИГНАЛОВ")
        print("=" * 60)

        print(f"Мω_изм.выб = {M_omega_vyb:.8f} × 10³")
        print(f"σ_изм.выб = {sigma_vyb:.8f}")

        normalized_signals = np.zeros_like(model_signals)

        # Статистики для каждого распределения
        distributions = [
            (0, "нрм", "ω_мод.нрм.иск"),
            (1, "пгс", "ω_мод.пгс.исх"),
            (2, "лпл", "ω_мод.лпл.иск"),
            (3, "гам", "ω_мод.гам.исх")
        ]

        for dist_code, dist_name, var_name in distributions:
            signal = model_signals[dist_code]
            M_signal = np.mean(signal)
            sigma_signal = np.std(signal, ddof=1)

            # Нормализация по формуле: (signal - M_signal) * (sigma_vyb / sigma_signal) + M_omega_vyb
            normalized_signal = (signal - M_signal) * (sigma_vyb / sigma_signal) + M_omega_vyb
            normalized_signals[dist_code] = normalized_signal

            print(f"\n{var_name} := ({var_name} – М_ω_мод.{dist_name}) · σ_изм.выб/σ_мод.{dist_name} + М_ω_изм.выб")
            print(f"М_ω_мод.{dist_name} := mean({var_name})")
            print(f"М_ω_мод.{dist_name} = {M_signal:.8f}")
            print(f"σ_мод.{dist_name} := stdev({var_name})")
            print(f"σ_мод.{dist_name} = {sigma_signal:.8f}")

            # Статистики после нормализации
            M_norm = np.mean(normalized_signal)
            sigma_norm = np.std(normalized_signal, ddof=1)
            print(f"После нормализации: М = {M_norm:.8f}, σ = {sigma_norm:.8f}")

        return normalized_signals

    def create_ordered_sequences(self, normalized_signals):
        """
        Формирование упорядоченных последовательностей для каждого распределения
        """
        print("\n" + "=" * 60)
        print("ФОРМИРОВАНИЕ УПОРЯДОЧЕННЫХ ПОСЛЕДОВАТЕЛЬНОСТЕЙ")
        print("=" * 60)

        ordered_sequences = {}

        distributions = [
            (0, "нрм", "ω_мод.нрм.уп"),
            (1, "пгс", "ω_мод.пгс.уп"),
            (2, "лпл", "ω_мод.лпл.уп"),
            (3, "гам", "ω_мод.гам.уп")
        ]

        for dist_code, dist_name, var_name in distributions:
            print(f"\n{var_name} :=")
            print("for il ∈ 0..L - 1")
            print("    for ik ∈ 0..K_инт - 1")
            print(f"    ω_исх_ik,il ← ω_мод.{dist_name}.исх_il·K_инт+ik")
            print("for il ∈ 0..L - 1")
            print("    ω_уп ← sort(ω_исх(il))")
            print("    ωω_уп ← ω_уп if il = 0")
            print("    ωω_уп ← augment(ωω_уп, ω_уп) otherwise")
            print("ωω_уп")

            # Реализация алгоритма
            signal = normalized_signals[dist_code]
            omega_omega_up = np.array([])

            # Преобразуем 1D массив в 2D: L строк по K_инт элементов
            signal_2d = signal.reshape(self.L, self.K_int)

            for il in range(self.L):
                # Берем отсчеты для текущего il
                omega_isk_il = signal_2d[il]

                # Сортируем отсчеты для текущего il
                omega_up = np.sort(omega_isk_il)

                # Добавляем в результирующий массив
                if il == 0:
                    omega_omega_up = omega_up
                else:
                    omega_omega_up = np.concatenate([omega_omega_up, omega_up])

            ordered_sequences[dist_name] = omega_omega_up

            # Выводим статистики упорядоченной последовательности
            M_ordered = np.mean(omega_omega_up)
            sigma_ordered = np.std(omega_omega_up, ddof=1)

            print(f"Размер упорядоченной последовательности: {len(omega_omega_up)}")
            print(f"Среднее: {M_ordered:.8f}")
            print(f"СКО: {sigma_ordered:.8f}")

        return ordered_sequences

    def calculate_ordered_statistics(self, ordered_sequences):
        """
        Вычисление статистик для упорядоченных последовательностей
        """
        print("\n" + "=" * 60)
        print("СТАТИСТИКИ УПОРЯДОЧЕННЫХ ПОСЛЕДОВАТЕЛЬНОСТЕЙ")
        print("=" * 60)

        distributions = [
            ("нрм", "ω_мод.нрм.уп"),
            ("пгс", "ω_мод.пгс.уп"),
            ("лпл", "ω_мод.лпл.уп"),
            ("гам", "ω_мод.гам.уп")
        ]

        for dist_name, var_name in distributions:
            signal = ordered_sequences[dist_name]
            M_signal = np.mean(signal)
            sigma_signal = np.std(signal, ddof=1)

            print(f"\n{var_name}")
            print(f"М_ω_мод.{dist_name}.уп = {M_signal:.8f}")
            print(f"σ_мод.{dist_name}.уп = {sigma_signal:.8f}")

            # Дополнительные статистики
            min_val = np.min(signal)
            max_val = np.max(signal)
            print(f"Min: {min_val:.8f}, Max: {max_val:.8f}")
            print(f"Диапазон: {max_val - min_val:.8f}")

    def pearson_criterion(self, ordered_sequences, M_omega_vyb, sigma_vyb):
        """
        4.1 Критерий согласия Пирсона
        """
        print("\n" + "=" * 60)
        print("4.1 КРИТЕРИЙ СОГЛАСИЯ ПИРСОНА")
        print("=" * 60)

        # Формула Стерджеса
        N_int = 1 + int(np.ceil(np.log(self.K_int)))
        print(f"Формула Стерджеса")
        print(f"N_инт := 1 + ceil(ln(K_инт))")
        print(f"N_инт = {N_int}")

        # Функции распределения
        def laplace_cdf(x, loc, scale):
            """Функция распределения Лапласа"""
            z = (x - loc) / scale
            return np.where(z <= 0, 0.5 * np.exp(z), 1 - 0.5 * np.exp(-z))

        distributions = [
            ("нрм", "нормальное", lambda x: stats.norm.cdf(x, M_omega_vyb, sigma_vyb)),
            ("пгс", "логистическое", lambda x: stats.logistic.cdf(x, M_omega_vyb, (np.sqrt(3) * sigma_vyb) / np.pi)),
            ("лпл", "лапласа", lambda x: laplace_cdf(x, M_omega_vyb, sigma_vyb / np.sqrt(2))),
            ("гам", "гамма", lambda x: stats.gamma.cdf(x, 2, loc=M_omega_vyb, scale=np.sqrt(sigma_vyb ** 2)))
        ]

        results = {}
        P_distributions = {}  # Храним вероятности для каждого распределения

        for dist_name, dist_full, cdf_func in distributions:
            print(f"\n--- {dist_full.upper()} распределение ---")

            signal = ordered_sequences[dist_name]

            # Определяем интервалы
            M_omega = np.max(signal)
            m_omega = np.min(signal)
            h = (M_omega - m_omega) / N_int

            print(f"h := (Mω_изм - mω_изм) / N_инт")
            print(f"h = ({M_omega:.6f} - {m_omega:.6f}) / {N_int} = {h:.6f}")

            # Границы интервалов
            a_int = np.zeros(N_int)
            b_int = np.zeros(N_int)

            for i in range(N_int):
                a_int[i] = m_omega + i * h
                b_int[i] = m_omega + (i + 1) * h

            # Вычисление теоретических вероятностей
            P_theoretical = np.zeros(N_int)

            print(f"P_{dist_name} :=")
            print("for in ∈ 0..N_инт - 1")

            for i in range(N_int):
                if dist_name == "нрм":
                    print(f"    pa_in ← pnorm(a_инт_in, Mω_изм.выб, σ_изм.выб)")
                    print(f"    pb_in ← pnorm(b_инт_in, Mω_изм.выб, σ_изм.выб)")
                elif dist_name == "пгс":
                    print(f"    pa_in ← plogis(a_инт_in, Mω_изм.выб, √3·σ_изм.выб/π)")
                    print(f"    pb_in ← plogis(b_инт_in, Mω_изм.выб, √3·σ_изм.выб/π)")
                elif dist_name == "лпл":
                    print(f"    pa_in ← [0.5·exp(σ_изм.выб(a_инт_in - Mω_изм.выб))] if a_инт_in ≤ Mω_изм.выб")
                    print(f"    pa_in ← [1 - 0.5·exp(-σ_изм.выб(a_инт_in - Mω_изм.выб))] otherwise")
                    print(f"    pb_in ← [0.5·exp(σ_изм.выб(b_инт_in - Mω_изм.выб))] if b_инт_in ≤ Mω_изм.выб")
                    print(f"    pb_in ← [1 - 0.5·exp(-σ_изм.выб(b_инт_in - Mω_изм.выб))] otherwise")
                elif dist_name == "гам":
                    print(f"    pa_in ← pgamma(a_инт_in, Mω_изм.выб, σ_изм.выб²)")
                    print(f"    pb_in ← pgamma(b_инт_in, Mω_изм.выб, σ_изм.выб²)")

                print(f"    p_in ← pb_in - pa_in")

                # Вычисляем вероятности
                pa = cdf_func(a_int[i])
                pb = cdf_func(b_int[i])
                P_theoretical[i] = pb - pa

            # Сохраняем вероятности для критерия Пирсона
            P_distributions[dist_name] = P_theoretical

            # Эмпирические частоты
            empirical_freq = np.zeros(N_int)
            for i in range(N_int):
                mask = (signal >= a_int[i]) & (signal < b_int[i])
                empirical_freq[i] = np.sum(mask)

            # Статистика хи-квадрат
            chi_square = 0
            for i in range(N_int):
                if P_theoretical[i] > 0:
                    expected = len(signal) * P_theoretical[i]
                    chi_square += (empirical_freq[i] - expected) ** 2 / expected

            # Степени свободы
            df = N_int - 1

            # P-value
            p_value = 1 - stats.chi2.cdf(chi_square, df)

            print(f"\nРезультаты для {dist_full}:")
            print(f"Статистика хи-квадрат: {chi_square:.6f}")
            print(f"Степени свободы: {df}")
            print(f"P-value: {p_value:.6f}")

            results[dist_name] = {
                'chi_square': chi_square,
                'df': df,
                'p_value': p_value,
                'intervals': N_int,
                'theoretical_probs': P_theoretical,
                'empirical_freq': empirical_freq
            }

        # Вычисление критерия Пирсона
        self.calculate_pearson_criterion(P_distributions, ordered_sequences, N_int)

        return results

    def pearson_criterion(self, ordered_sequences, M_omega_vyb, sigma_vyb):
        """
        4.1 Критерий согласия Пирсона
        """
        print("\n" + "=" * 60)
        print("4.1 КРИТЕРИЙ СОГЛАСИЯ ПИРСОНА")
        print("=" * 60)

        # Формула Стерджеса
        N_int = 1 + int(np.ceil(np.log(self.K_int)))
        print(f"Формула Стерджеса")
        print(f"N_инт := 1 + ceil(ln(K_инт))")
        print(f"N_инт = {N_int}")

        # Функция построения гистограммы
        def build_histogram(signal, N_int):
            """Построение гистограммы"""
            M_omega = np.max(signal)
            m_omega = np.min(signal)
            h = (M_omega - m_omega) / N_int

            # Границы интервалов
            a_int = np.zeros(N_int)
            b_int = np.zeros(N_int)
            for i in range(N_int):
                a_int[i] = m_omega + i * h
                b_int[i] = m_omega + (i + 1) * h

            # Частоты гистограммы
            N_hist = np.zeros(N_int)

            print("Гист :=")
            print("for in ∈ 0..N_инт - 1")
            print("    N_гист_in ← 0")
            print("    for ik ∈ 0..K_инт - 1")
            print("    N_гист_in ← N_гист_in + 1 if a_инт_in ≤ ω_изм.исх_ik < b_инт_in")
            print("N_гист")

            for i in range(N_int):
                count = 0
                for value in signal:
                    if a_int[i] <= value < b_int[i]:
                        count += 1
                N_hist[i] = count

            return a_int, b_int, N_hist, h

        # Функции распределения
        def laplace_cdf(x, loc, scale):
            """Функция распределения Лапласа"""
            z = (x - loc) / scale
            return np.where(z <= 0, 0.5 * np.exp(z), 1 - 0.5 * np.exp(-z))

        distributions = [
            ("нрм", "нормальное", lambda x: stats.norm.cdf(x, M_omega_vyb, sigma_vyb)),
            ("пгс", "логистическое", lambda x: stats.logistic.cdf(x, M_omega_vyb, (np.sqrt(3) * sigma_vyb) / np.pi)),
            ("лпл", "лапласа", lambda x: laplace_cdf(x, M_omega_vyb, sigma_vyb / np.sqrt(2))),
            ("гам", "гамма", lambda x: stats.gamma.cdf(x, 2, loc=M_omega_vyb, scale=np.sqrt(sigma_vyb ** 2)))
        ]

        results = {}
        P_distributions = {}
        k_pearson_results = {}

        for dist_name, dist_full, cdf_func in distributions:
            print(f"\n--- {dist_full.upper()} распределение ---")

            signal = ordered_sequences[dist_name]

            # Строим гистограмму
            a_int, b_int, N_hist, h = build_histogram(signal, N_int)

            print(f"h = {h:.6f}")
            print(f"Интервалы гистограммы:")
            for i in range(N_int):
                print(f"  [{a_int[i]:.4f}, {b_int[i]:.4f}]: N_гист = {N_hist[i]}")

            # Вычисление теоретических вероятностей
            P_theoretical = np.zeros(N_int)

            print(f"\nP_{dist_name} :=")
            print("for in ∈ 0..N_инт - 1")

            for i in range(N_int):
                if dist_name == "нрм":
                    print(f"    pa_in ← pnorm(a_инт_in, Mω_изм.выб, σ_изм.выб)")
                    print(f"    pb_in ← pnorm(b_инт_in, Mω_изм.выб, σ_изм.выб)")
                elif dist_name == "пгс":
                    print(f"    pa_in ← plogis(a_инт_in, Mω_изм.выб, √3·σ_изм.выб/π)")
                    print(f"    pb_in ← plogis(b_инт_in, Mω_изм.выб, √3·σ_изм.выб/π)")
                elif dist_name == "лпл":
                    print(f"    pa_in ← [0.5·exp(σ_изм.выб(a_инт_in - Mω_изм.выб))] if a_инт_in ≤ Mω_изм.выб")
                    print(f"    pa_in ← [1 - 0.5·exp(-σ_изм.выб(a_инт_in - Mω_изм.выб))] otherwise")
                    print(f"    pb_in ← [0.5·exp(σ_изм.выб(b_инт_in - Mω_изм.выб))] if b_инт_in ≤ Mω_изм.выб")
                    print(f"    pb_in ← [1 - 0.5·exp(-σ_изм.выб(b_инт_in - Mω_изм.выб))] otherwise")
                elif dist_name == "гам":
                    print(f"    pa_in ← pgamma(a_инт_in, Mω_изм.выб, σ_изм.выб²)")
                    print(f"    pb_in ← pgamma(b_инт_in, Mω_изм.выб, σ_изм.выб²)")

                print(f"    p_in ← pb_in - pa_in")

                # Вычисляем вероятности
                pa = cdf_func(a_int[i])
                pb = cdf_func(b_int[i])
                P_theoretical[i] = pb - pa

            P_distributions[dist_name] = P_theoretical

            # Вычисление критерия Пирсона
            k_pearson = self.calculate_pearson_for_distribution(N_hist, P_theoretical, len(signal), N_int, dist_name)
            k_pearson_results[dist_name] = k_pearson

            # Статистика хи-квадрат
            chi_square = 0
            for i in range(N_int):
                if P_theoretical[i] > 0:
                    expected = len(signal) * P_theoretical[i]
                    chi_square += (N_hist[i] - expected) ** 2 / expected

            df = N_int - 1
            p_value = 1 - stats.chi2.cdf(chi_square, df)

            print(f"\nРезультаты для {dist_full}:")
            print(f"Статистика хи-квадрат: {chi_square:.6f}")
            print(f"Степени свободы: {df}")
            print(f"P-value: {p_value:.6f}")
            print(f"Критерий Пирсона: {k_pearson:.6f}")

            results[dist_name] = {
                'chi_square': chi_square,
                'df': df,
                'p_value': p_value,
                'k_pearson': k_pearson,
                'intervals': N_int,
                'theoretical_probs': P_theoretical,
                'empirical_freq': N_hist
            }

        # Итоговый вывод критерия Пирсона
        self.print_final_pearson_results(k_pearson_results)

        return results

    def calculate_pearson_for_distribution(self, N_hist, P_theoretical, signal_length, N_int, dist_name):
        """
        Вычисление критерия Пирсона для одного распределения
        """
        k_pearson = 0
        valid_intervals = 0

        for i in range(N_int):
            if P_theoretical[i] > 0:
                expected = (signal_length // self.L) * P_theoretical[i]  # K_инт * P
                if expected > 0:
                    k_pearson += (N_hist[i] - expected) ** 2 / expected
                    valid_intervals += 1

        if valid_intervals > 0:
            k_pearson = k_pearson / valid_intervals

        return k_pearson

    def print_final_pearson_results(self, k_pearson_results):
        """
        Вывод итоговых результатов критерия Пирсона
        """
        print("\n" + "=" * 60)
        print("КРИТЕРИЙ ПИРСОНА - ИТОГИ")
        print("=" * 60)

        print("K_Пирсона := for in ∈ 0..N_сиг - 1")

        dist_mapping = {
            "нрм": ("k_Пирсона_0", "P_нрм"),
            "пгс": ("k_Пирсона_1", "P_лгс"),
            "лпл": ("k_Пирсона_2", "P_лпл"),
            "гам": ("k_Пирсона_3", "P_гам")
        }

        dist_names = {
            "нрм": "нормальное",
            "пгс": "логистическое",
            "лпл": "лапласа",
            "гам": "гамма"
        }

        for dist_name, (k_name, p_name) in dist_mapping.items():
            if dist_name in k_pearson_results:
                k_value = k_pearson_results[dist_name]
                print(f"{k_name} ← (1/N_инт) · Σ[ (Γ_пст - K_инт·{p_name}_in)² / (K_инт·{p_name}_in) ]")
                print(f"{k_name} = {k_value:.6f}")

        print(f"\nИтоговые значения критерия Пирсона:")
        for dist_name, dist_full in dist_names.items():
            if dist_name in k_pearson_results:
                print(f"k_Пирсона ({dist_full}): {k_pearson_results[dist_name]:.6f}")

        # Определение наилучшего распределения
        if k_pearson_results:
            best_dist = min(k_pearson_results.items(), key=lambda x: x[1])
            print(f"\nНаилучшее соответствие: {dist_names[best_dist[0]]} распределение")
            print(f"Значение критерия: {best_dist[1]:.6f}")

    def cramer_von_mises_criterion(self, ordered_sequences, observed_signal_sorted, M_omega_vyb, sigma_vyb):
        """
        4.2 Критерий Крамера-фон Мизеса
        """
        print("\n" + "=" * 60)
        print("4.2 КРИТЕРИЙ КРАМЕРА-ФОН МИЗЕСА")
        print("=" * 60)

        # Функция для распределения Лапласа
        def laplace_cdf(x, loc, scale):
            z = (x - loc) / scale
            return np.where(z <= 0, 0.5 * np.exp(z), 1 - 0.5 * np.exp(-z))

        # Вычисление plap_ИЗМ для наблюдаемого сигнала
        plap_izm = np.zeros(len(observed_signal_sorted))
        for ik in range(len(observed_signal_sorted)):
            if observed_signal_sorted[ik] <= self.omega_0_0:
                plap_izm[ik] = 0.5 * np.exp(self.sigma_0_0 * (observed_signal_sorted[ik] - self.omega_0_0))
            else:
                plap_izm[ik] = 1 - 0.5 * np.exp(-self.sigma_0_0 * (observed_signal_sorted[ik] - self.omega_0_0))

        print("plap_ИЗМ,ik :=")
        print("0.5·exp[σ₀(ω_ИЗМ,УП,ik - ω₀)] if ω_ИЗМ,УП,ik ≤ ω₀")
        print("1 - 0.5·exp[-σ₀(ω_ИЗМ,УП,ik - ω₀)] otherwise")

        # Вычисление plap_МОД для модельных сигналов
        plap_mod = {}
        distributions = ["нрм", "пгс", "лпл"]

        for dist_name in distributions:
            signal = ordered_sequences[dist_name]
            plap_mod[dist_name] = np.zeros(len(signal))

            for ik in range(len(signal)):
                if signal[ik] <= M_omega_vyb:
                    if dist_name == "лпл":
                        plap_mod[dist_name][ik] = 0.5 * np.exp(
                            (sigma_vyb / np.sqrt(2)) * (signal[ik] - M_omega_vyb)
                        )
                    else:
                        # Для других распределений используем соответствующие CDF
                        if dist_name == "нрм":
                            plap_mod[dist_name][ik] = stats.norm.cdf(signal[ik], M_omega_vyb, sigma_vyb)
                        elif dist_name == "пгс":
                            plap_mod[dist_name][ik] = stats.logistic.cdf(signal[ik], M_omega_vyb,
                                                                         (np.sqrt(3) * sigma_vyb) / np.pi)
                else:
                    if dist_name == "лпл":
                        plap_mod[dist_name][ik] = 1 - 0.5 * np.exp(
                            -(sigma_vyb / np.sqrt(2)) * (signal[ik] - M_omega_vyb)
                        )
                    else:
                        if dist_name == "нрм":
                            plap_mod[dist_name][ik] = stats.norm.cdf(signal[ik], M_omega_vyb, sigma_vyb)
                        elif dist_name == "пгс":
                            plap_mod[dist_name][ik] = stats.logistic.cdf(signal[ik], M_omega_vyb,
                                                                         (np.sqrt(3) * sigma_vyb) / np.pi)

        print("\nplap_МОД,ik,il :=")
        print("0.5·exp[σ_ИЗМ,ВЫб/√2(ω_МОД,ЛПЛ,УП,ik,il - Мω_изм,выб)] if ω_ИЗМ,УП,ik ≤ Мω_изм,выб")
        print("1 - 0.5·exp[σ_ИЗМ,ВЫб/√2(ω_МОД,ЛПЛ,УП,ik,il - Мω_изм,выб)] otherwise")

        # Вычисление критерия KM
        kfm_results = {}
        K_mrr = min(len(observed_signal_sorted), len(ordered_sequences["нрм"]))

        print(f"\nKM :=")

        if self.in0 == 0:  # Нормальное распределение
            print("if in0 = 0")

            # Для нормального наблюдаемого распределения
            p_obs = stats.norm.cdf(observed_signal_sorted[:K_mrr], self.omega_0_0, self.sigma_0_0)

            kfm_0 = np.mean((p_obs - stats.norm.cdf(ordered_sequences["нрм"][:K_mrr], M_omega_vyb, sigma_vyb)) ** 2)
            kfm_1 = np.mean((p_obs - stats.logistic.cdf(ordered_sequences["пгс"][:K_mrr], M_omega_vyb,
                                                        (np.sqrt(3) * sigma_vyb) / np.pi)) ** 2)
            kfm_2 = np.mean((p_obs - plap_mod["лпл"][:K_mrr]) ** 2)

            print("kfm_0 ← (1/K_mrr) Σ[pnorm(ω_изм.уп_ik,ω₀) - pnorm(ω_мод.нрм.уп_ik,Мω_изм.выб,σ_изм.выб)]²")
            print("kfm_1 ← (1/K_mrr) Σ[pnorm(ω_изм.уп_ik,ω₀) - plogis(ω_мод.пгс.уп_ik,Мω_изм.выб,σ_изм.выб)]²")
            print("kfm_2 ← (1/K_mrr) Σ[pnorm(ω_изм.уп_ik,ω₀) - plap_мод_ik]²")

        elif self.in0 == 1:  # Логистическое распределение
            print("if in0 = 1")

            # Для логистического наблюдаемого распределения
            p_obs = stats.logistic.cdf(observed_signal_sorted[:K_mrr], self.omega_0_0,
                                       (np.sqrt(3) * self.sigma_0_0) / np.pi)

            kfm_0 = np.mean((p_obs - stats.norm.cdf(ordered_sequences["нрм"][:K_mrr], M_omega_vyb, sigma_vyb)) ** 2)
            kfm_1 = np.mean((p_obs - stats.logistic.cdf(ordered_sequences["пгс"][:K_mrr], M_omega_vyb,
                                                        (np.sqrt(3) * sigma_vyb) / np.pi)) ** 2)
            kfm_2 = np.mean((p_obs - plap_mod["лпл"][:K_mrr]) ** 2)

            print("kfm_0 ← (1/K_mrr) Σ[plogis(ω_изм.уп_ik,ω₀) - pnorm(ω_мод.нрм.уп_ik,Мω_изм.выб,σ_изм.выб)]²")
            print("kfm_1 ← (1/K_mrr) Σ[plogis(ω_изм.уп_ik,ω₀) - plogis(ω_мод.пгс.уп_ik,Мω_изм.выб,σ_изм.выб)]²")
            print("kfm_2 ← (1/K_mrr) Σ[plogis(ω_изм.уп_ik,ω₀) - plap_мод_ik]²")

        elif self.in0 == 2:  # Лапласа распределение
            print("if in0 = 2")

            # Для лапласовского наблюдаемого распределения
            p_obs = plap_izm[:K_mrr]

            kfm_0 = np.mean((p_obs - stats.norm.cdf(ordered_sequences["нрм"][:K_mrr], M_omega_vyb, sigma_vyb)) ** 2)
            kfm_1 = np.mean((p_obs - stats.logistic.cdf(ordered_sequences["пгс"][:K_mrr], M_omega_vyb,
                                                        (np.sqrt(3) * sigma_vyb) / np.pi)) ** 2)
            kfm_2 = np.mean((p_obs - plap_mod["лпл"][:K_mrr]) ** 2)

            print("kfm_0 ← (1/K_mrr) Σ[plap_изм_ik - pnorm(ω_мод.нрм.уп_ik,Мω_изм.выб,σ_изм.выб)]²")
            print("kfm_1 ← (1/K_mrr) Σ[plap_изм_ik - plogis(ω_мод.пгс.уп_ik,Мω_изм.выб,σ_изм.выб)]²")
            print("kfm_2 ← (1/K_mrr) Σ[plap_изм_ik - plap_мод_ik]²")

        kfm_results = {
            'kfm_0': kfm_0,
            'kfm_1': kfm_1,
            'kfm_2': kfm_2
        }

        print(f"\nРезультаты критерия Крамера-фон Мизеса:")
        print(f"kfm_0 (нормальное): {kfm_0:.8f}")
        print(f"kfm_1 (логистическое): {kfm_1:.8f}")
        print(f"kfm_2 (лапласа): {kfm_2:.8f}")

        # Определение наилучшего распределения
        best_idx = np.argmin([kfm_0, kfm_1, kfm_2])
        dist_names = ["нормальное", "логистическое", "лапласа"]
        print(
            f"\nНаилучшее соответствие: {dist_names[best_idx]} распределение (kfm = {[kfm_0, kfm_1, kfm_2][best_idx]:.8f})")

        return kfm_results

    def correlation_criterion(self, ordered_sequences, observed_signal_sorted, M_omega_vyb, sigma_vyb):
        """
        4.3 Корреляционный критерий
        """
        print("\n" + "=" * 60)
        print("4.3 КОРРЕЛЯЦИОННЫЙ КРИТЕРИЙ")
        print("=" * 60)

        print("Cor_ИЗМ := for in ∈ 0..N_CHT - 1")

        # Исправляем размеры - наблюдаемый сигнал имеет размер K_int, а модельные - L * K_int
        # Берем только первые K_int отсчетов наблюдаемого сигнала для сравнения
        observed_signal_short = observed_signal_sorted[:self.K_int]

        cor_results = {}
        distributions = [
            ("нрм", "ω_МОД.НРМ.УП"),
            ("пгс", "ω_МОД.ЛГС.УП"),
            ("лпл", "ω_МОД.ЛПЛ.УП"),
            ("гам", "ω_МОД.ГАМ.УП")
        ]

        for i, (dist_name, dist_var) in enumerate(distributions):
            if dist_name in ordered_sequences:
                # Преобразуем модельную последовательность в 2D (L x K_инт)
                model_signal = ordered_sequences[dist_name]
                model_2d = model_signal.reshape(self.L, self.K_int)

                # Вычисляем числитель: (L-1) * Σ_il Σ_ik (ω_изм.уп_ik - ω_мод.уп_ik,il)²
                numerator_sum = 0
                for il in range(self.L):
                    for ik in range(self.K_int):
                        diff = observed_signal_short[ik] - model_2d[il, ik]
                        numerator_sum += diff ** 2

                numerator = (self.L - 1) * numerator_sum

                # Вычисляем знаменатель: L * Σ_ik (ω_изм.уп_ik)²
                denominator_sum = np.sum(observed_signal_short ** 2)
                denominator = self.L * denominator_sum

                # Вычисляем критерий
                cor_value = numerator / denominator if denominator != 0 else float('inf')
                cor_results[dist_name] = cor_value

                print(
                    f"\ncor_{i} ← [ (L-1) · Σ_il Σ_ik (ω_ИЗМ.УП_ik - {dist_var}_ik,il)² ] / [ L · Σ_ik (ω_ИЗМ.УП_ik)² ]")
                print(f"cor_{i} = {cor_value:.8f}")

                # Детали вычислений
                print(f"  Числитель: ({self.L}-1) × {numerator_sum:.6f} = {numerator:.6f}")
                print(f"  Знаменатель: {self.L} × {denominator_sum:.6f} = {denominator:.6f}")

        # Итоговый вывод
        print(f"\nИтоговые значения корреляционного критерия:")
        dist_names = {
            "нрм": "нормальное",
            "пгс": "логистическое",
            "лпл": "лапласа",
            "гам": "гамма"
        }

        for dist_name, cor_value in cor_results.items():
            print(f"cor ({dist_names[dist_name]}): {cor_value:.8f}")

        # Определение наилучшего распределения (минимальное значение)
        if cor_results:
            best_dist = min(cor_results.items(), key=lambda x: x[1])
            print(f"\nНаилучшее соответствие: {dist_names[best_dist[0]]} распределение")
            print(f"Значение критерия: {best_dist[1]:.8f}")

        return cor_results

# ИСПОЛЬЗОВАНИЕ
if __name__ == "__main__":
    # Создаем процессор сигналов
    processor = SignalProcessor()

    # 1. Выводим исходные данные
    processor.print_initial_parameters()

    # 2. Строим график наблюдаемого сигнала ω_изм.исх_ik
    observed_signal = processor.plot_omega_izm_isk()

    # 3. Строим график упорядоченной последовательности ω_изм.уч
    sorted_signal = processor.plot_omega_izm_uch(observed_signal)

    # 4. 3.1 Расчет статистик выборки
    M_omega_vyb, sigma_vyb, variance_vyb = processor.calculate_sample_statistics(observed_signal)

    # 5. 3.2 Формирование модельных сигналов
    model_signals = processor.generate_model_signals(sigma_vyb)

    # 6. Вычисление статистик модельных сигналов
    model_stats = processor.calculate_model_statistics(model_signals)

    # 7. Нормализация модельных сигналов
    normalized_signals = processor.normalize_model_signals(model_signals, M_omega_vyb, sigma_vyb)

    # 7. Формирование упорядоченных последовательностей
    ordered_sequences = processor.create_ordered_sequences(normalized_signals)

    # 8. Статистики упорядоченных последовательностей
    processor.calculate_ordered_statistics(ordered_sequences)

    # 9. Критерий согласия Пирсона
    pearson_results = processor.pearson_criterion(ordered_sequences, M_omega_vyb, sigma_vyb)

    # 10. Критерий Крамера-фон Мизеса
    observed_signal_sorted = np.sort(observed_signal)
    cramer_results = processor.cramer_von_mises_criterion(ordered_sequences, observed_signal_sorted, M_omega_vyb, sigma_vyb)

    # 11. Корреляционный критерий
    correlation_results = processor.correlation_criterion(ordered_sequences, observed_signal_sorted, M_omega_vyb,
                                                          sigma_vyb)