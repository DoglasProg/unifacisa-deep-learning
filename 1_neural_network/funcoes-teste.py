def calculo_de_tempo_em_execucao(start: float = time()) -> float:
    """Método que irá calcular o tempo de execução a partir de um start

    Args:
        start (float, optional): tempo decorrido após o start. Defaults to time().

    Returns:
        float: tempo decorrido de execução.
    """
    print ('Tempo gasto: %s em segundos \n' % str(time() - start))
    return time() - start