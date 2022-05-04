class Generator:
    __list_adv = [
        '还真是',
        '可真是',
        '真是极其的',
    ]
    __list_adj = [
        '出色',
        '棒',
        '优秀',
        '值得夸奖',
        '令人赞赏',
    ]

    def __combine(self):
        comments = []
        for adv in self.__list_adv:
            for adj in self.__list_adj:
                comments.append(adv + adj)
        return comments

    def gen(self):
        comments = self.__combine()
        new_comments = []
        for comment in comments:
            c = comment + '！'
            for i in range(0, 5):
                c += comment + '！'
                new_comments.append(c)
        return comments + new_comments
