
def parce_PSPLib_graph(file_from: str, file_to: str)-> None:
    str_to_print = ""
    with open(file_from, 'r') as f:
        lines = f.readlines()
        state = 'search'
        for line in lines:
            if line == 'PRECEDENCE RELATIONS:\n':
                state = 'header'
                continue
            if state == 'header':
                state = 'data'
                continue
            if state == 'data' and line[0] == '*':
                break
            if state == 'data':
                data = line.split()
                if len(data) > 3:
                    fr_id = str(int(data[0]) - 1)
                    to_ids = [str(int(data[i])-1) for i in range(3, len(data))]
                    str_to_print += fr_id + ':' + ','.join(to_ids) + '\n'
    with open(file_to, 'w') as f:
        f.write(str_to_print)
        f.close()


if __name__ == '__main__':
    Ns = [60, 120]
    ks = [48, 60]
    ds = [10, 10]
    for round in range(len(Ns)):
        N = Ns[round]
        k = ks[round]
        d = ds[round]
        for i in range(1, k+1):
            for j in range(1, d+1):
                file_from = f'./Data/PSPLib/j{N}.sm/j{N}{i}_{j}.sm'
                file_to = f'./Data/PrecedenceGraphs/parsedPSPLib/graph_{N+2}_{(i-1) * d + j - 1}.txt'
                parce_PSPLib_graph(file_from, file_to)
