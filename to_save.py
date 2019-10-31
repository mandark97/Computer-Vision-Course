# if max_mean < 15:
#             if len(values) == 3:
#                 detected_x = np.array(values)[:, 0]
#                 manevra = x_coords.copy()

#                 for x in detected_x:
#                     ceva = np.argmin([abs(x_coord - x)
#                                         for x_coord in manevra])
#                     manevra = np.delete(manevra, ceva)

#                 correct_answer = list(x_coords).index(manevra[0])

#             else:
#                 # print(
#                 #     f"question {offset + index + 1} meh meh {len(values)}")
#                 correct_answer = answer

#         else:
#             answer_x = values[answer][0]
#             correct_answer = np.argmin(
#                 [abs(x_coord - answer_x) for x_coord in x_coords])
