from stabtab import Tableau


def main():
    tab = Tableau(1)
    tab.apply_h(0)
    tab.measure_z(0)
    print(tab)


if __name__ == "__main__":
    main()
