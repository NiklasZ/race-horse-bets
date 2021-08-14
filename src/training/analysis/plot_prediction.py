# import numpy as np
# from matplotlib import pyplot as plt
#
# # TODO finish
# def get_race_id(full_race: np.ndarray) -> str:
#     norm_data = np.vstack((norm_training, norm_validation, norm_test))
#     index = np.where(np.all(norm_data == full_race, axis=2))[0][0]
#     print(index)
#     return data.iloc[index * config['target_race_length']]['refid']
#
#
# markers = ['P', 's', 'p', 'X', '*', 'D']
# colours = ['black', 'aqua', 'blue', 'brown', 'coral', 'goldenrod', 'green', 'indigo', 'lime', 'magenta', 'red',
#            'purple', 'turquoise', 'yellow', 'sienna', 'khaki']
#
#
# def plot_predictions(test_series, test_input, test_label):
#     [input_cycles, input_pools] = test_input.shape
#
#     # Plot the existing time series and future
#     for input_column, series_column, colour in (zip(test_input.T, test_series.T, colours)):
#         plt.plot(range(0, input_cycles), input_column, marker='.', color=colour)
#         future_values = np.insert(series_column[len(input_column):], 0, input_column[-1])
#         plt.plot(range(input_cycles - 1, input_cycles - 1 + len(future_values)), future_values, marker='.',
#                  color=colour,
#                  alpha=0.3, linewidth=2)
#
#     # Plot model predictions
#     for model, marker in zip(models, markers):
#         predictions = np.squeeze(model(test_input))
#         model_name = model.__class__.__name__
#         label_set = False
#         for p, colour in zip(predictions, colours):
#             label = None if label_set else f'{model_name}'
#             plt.plot(len(test_series) - 1, p, marker, linestyle='', color=colour,
#                      label=label)
#             label_set = True
#
#     plt.ylabel('Bet Pool - normalised')
#     plt.xlabel(f'Cycles (30 seconds each)')
#     refid = get_race_id(test_series)
#     plt.title(f"Race {refid} - Model Predictions {config['cycles_into_the_future']} into the future")
#     plt.legend()
#     plt.show()
#
#
# if __name__ == '__main__':
#     plot_predictions(norm_test[15], test_inputs[15], test_labels[15])
