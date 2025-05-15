def rle_encode(ac_coefficients):
    rle = []
    run = 0  # Счетчик нулей

    for coeff in ac_coefficients:
        if coeff == 0:
            run += 1
        else:
            # Добавляем пару (run, value)
            rle.append((run, coeff))
            run = 0

    # Добавляем EOB, если блок не пустой
    if len(ac_coefficients) > 0:
        rle.append((0, 0))

    return rle


def rle_decode(rle, block_size=64):
    ac_coefficients = []

    for run, value in rle:
        if run == 0 and value == 0:  # EOB
            # Заполняем оставшееся место нулями до размера блока
            ac_coefficients.extend([0] * (block_size - len(ac_coefficients)))
            break
        # Добавляем run нулей
        ac_coefficients.extend([0] * run)
        # Добавляем ненулевое значение
        ac_coefficients.append(value)

    return ac_coefficients