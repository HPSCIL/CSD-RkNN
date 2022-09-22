import experiments
from common.fig import plot_dual_distribution, plot_single_distribution

if __name__ == '__main__':
    # effect of data size on Mono-RkNN
    time_cost, io_cost = experiments.BenchmarkExperiments.evaluate_effect_of_data_size_on_MonoRkNN(k=10)
    print('#effect of data size on Mono-RkNN (k=10)')
    print('time_cost=' + str(time_cost))
    print('io_cost=' + str(io_cost))
    plot_dual_distribution(time_cost, 'Effect-of-data-size-on-MonoRkNN-time-cost(k=10)')
    plot_dual_distribution(io_cost, 'Effect-of-data-size-on-MonoRkNN-io-cost(k=10)')
    time_cost, io_cost = experiments.BenchmarkExperiments.evaluate_effect_of_data_size_on_MonoRkNN(k=1000)
    plot_dual_distribution(time_cost, 'Effect-of-data-size-on-MonoRkNN-time-cost(k=1000)')
    plot_dual_distribution(io_cost, 'Effect-of-data-size-on-MonoRkNN-io-cost(k=1000)')
    print('#effect of data size on Mono-RkNN (k=1000)')
    print('time_cost=' + str(time_cost))
    print('io_cost=' + str(io_cost))

    # effect of data size on Bi-RkNN
    time_cost, io_cost = experiments.BenchmarkExperiments.evaluate_effect_of_data_size_on_BiRkNN(k=10)
    print('#effect of data size on Bi-RkNN (k=10)')
    print('time_cost=' + str(time_cost))
    print('io_cost=' + str(io_cost))
    plot_dual_distribution(time_cost, 'Effect-of-data-size-on-BiRkNN-time-cost(k=10)')
    plot_dual_distribution(io_cost, 'Effect-of-data-size-on-BiRkNN-io-cost(k=10)')
    time_cost, io_cost = experiments.BenchmarkExperiments.evaluate_effect_of_data_size_on_BiRkNN(k=1000)
    print('#effect of data size on Bi-RkNN (k=1000)')
    print('time_cost=' + str(time_cost))
    print('io_cost=' + str(io_cost))
    plot_dual_distribution(time_cost, 'Effect-of-data-size-on-BiRkNN-time-cost(k=1000)')
    plot_dual_distribution(io_cost, 'Effect-of-data-size-on-BiRkNN-io-cost(k=1000)')

    # effect of k on Mono-RkNN
    time_cost, io_cost = experiments.BenchmarkExperiments.evaluate_effect_of_k_on_MonoRkNN(distribution='Synthetic')
    plot_dual_distribution(time_cost, 'Effect-of-k-on-MonoRkNN-time-cost(Synthetic)')
    plot_dual_distribution(io_cost, 'Effect-of-k-on-MonoRkNN-io-cost(Synthetic)')
    print('#effect of k on Mono-RkNN (Synthetic)')
    print('time_cost=' + str(time_cost))
    print('io_cost=' + str(io_cost))
    time_cost, io_cost = experiments.BenchmarkExperiments.evaluate_effect_of_k_on_MonoRkNN(distribution='Real')
    print('#effect of k on Mono-RkNN (Real)')
    print('time_cost=' + str(time_cost))
    print('io_cost=' + str(io_cost))
    plot_single_distribution(time_cost, 'Effect-of-k-on-MonoRkNN-time-cost(Real)')
    plot_single_distribution(io_cost, 'Effect-of-k-on-MonoRkNN-io-cost(Real)')

    # effect of k on Bi-RkNN
    time_cost, io_cost = experiments.BenchmarkExperiments.evaluate_effect_of_k_on_BiRkNN(distribution='Synthetic')
    print('#effect of k on Bi-RkNN (Synthetic)')
    print('time_cost=' + str(time_cost))
    print('io_cost=' + str(io_cost))
    plot_dual_distribution(time_cost, 'Effect-of-k-on-BiRkNN-time-cost(Synthetic)')
    plot_dual_distribution(io_cost, 'Effect-of-k-on-BiRkNN-io-cost(Synthetic)')

    time_cost, io_cost = experiments.BenchmarkExperiments.evaluate_effect_of_k_on_BiRkNN(distribution='Real')
    print('#effect of k on Bi-RkNN (Real)')
    print('time_cost=' + str(time_cost))
    print('io_cost=' + str(io_cost))
    plot_single_distribution(time_cost, 'Effect-of-k-on-BiRkNN-time-cost(Real)')
    plot_single_distribution(io_cost, 'Effect-of-k-on-BiRkNN-io-cost(Real)')

    # effect of number of users relative to number of facilities
    time_cost, io_cost = experiments.BenchmarkExperiments.evaluate_effect_of_user_num_relative_to_facility_num(k=10)
    print('#effect of user relative to facility on Bi-RkNN (k=10)')
    print('time_cost=' + str(time_cost))
    print('io_cost=' + str(io_cost))
    plot_dual_distribution(time_cost, 'Effect-of-user-relative-to-facility-on-BiRkNN-time-cost(k=10)')
    plot_dual_distribution(io_cost, 'Effect-of-user-relative-to-facility-on-BiRkNN-io-cost(k=10)')
    time_cost, io_cost = experiments.BenchmarkExperiments.evaluate_effect_of_user_num_relative_to_facility_num(k=1000)
    print('#effect of user relative to facility on Bi-RkNN (k=1000)')
    print('time_cost=' + str(time_cost))
    print('io_cost=' + str(io_cost))
    plot_dual_distribution(time_cost, 'Effect-of-user-relative-to-facility-on-BiRkNN-time-cost(k=1000)')
    plot_dual_distribution(io_cost, 'Effect-of-user-relative-to-facility-on-BiRkNN-io-cost(k=1000)')

    # effect of data distribution
    time_cost, io_cost = experiments.BenchmarkExperiments.evaluate_effect_of_data_distribution(k=10)
    print('#effect of data distribution on Bi-RkNN (k=10)')
    print('time_cost=' + str(time_cost))
    print('io_cost=' + str(io_cost))
    plot_single_distribution(time_cost, 'Effect-of-distribution-on-BiRkNN-time-cost(k=10)')
    plot_single_distribution(io_cost, 'Effect-of-distribution-on-BiRkNN-io-cost(k=10)')
    time_cost, io_cost = experiments.BenchmarkExperiments.evaluate_effect_of_data_distribution(k=1000)
    print('#effect of data distribution on Bi-RkNN (k=1000)')
    print('time_cost=' + str(time_cost))
    print('io_cost=' + str(io_cost))
    plot_single_distribution(time_cost, 'Effect-of-distribution-on-BiRkNN-time-cost(k=1000)')
    plot_single_distribution(io_cost, 'Effect-of-distribution-on-BiRkNN-io-cost(k=1000)')

    # effect of k on RkNN for restaurant
    time_cost, io_cost = experiments.CaseStudyExperiments.evaluate_RkNN_for_restaurant()
    print('#effect of k on RkNN for restaurant')
    print('time_cost=' + str(time_cost))
    print('io_cost=' + str(io_cost))
    plot_single_distribution(time_cost, 'Time-cost-of-RkNN-for-restaurant')
    plot_single_distribution(io_cost, 'IO-cost-of-RkNN-for-restaurant')

    # effect of k on RkNN for mall
    time_cost, io_cost = experiments.CaseStudyExperiments.evaluate_RkNN_for_mall()
    print('#effect of k on RkNN for mall')
    print('time_cost=' + str(time_cost))
    print('io_cost=' + str(io_cost))
    plot_single_distribution(time_cost, 'Time-cost-of-RkNN-for-mall')
    plot_single_distribution(io_cost, 'IO-cost-of-RkNN-for-mall')

    # effect of k on RkNN for hospital
    time_cost, io_cost = experiments.CaseStudyExperiments.evaluate_RkNN_for_hospital()
    print('#effect of k on RkNN for hospital')
    print('time_cost=' + str(time_cost))
    print('io_cost=' + str(io_cost))
    plot_single_distribution(time_cost, 'Time-cost-of-RkNN-for-hospital')
    plot_single_distribution(io_cost, 'IO-cost-of-RkNN-for-hospital')

    # effect of k on RkNN for school
    time_cost, io_cost = experiments.CaseStudyExperiments.evaluate_RkNN_for_school()
    print('#effect of k on RkNN for school')
    print('time_cost=' + str(time_cost))
    print('io_cost=' + str(io_cost))
    plot_single_distribution(time_cost, 'Time-cost-of-RkNN-for-school')
    plot_single_distribution(io_cost, 'IO-cost-of-RkNN-for-school')
