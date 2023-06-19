def crust():
    print("70km")

    def mantle():
        print("2900km")

        def core():
            print("5300km")
            return mantle()

        return core

    return mantle


drill = crust
drill = drill()
