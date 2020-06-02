
class trie:
    def __init__(self):
        self.root = {}
        self.word_end = -1

    def insert(self, word):
        current_node = self.root
        for ch in word:
            if ch not in current_node:
                current_node[ch] = {}
            current_node = current_node[ch]
        current_node[self.word_end] = True
    
    def batch_insert(self, words):
        for word in words:
            self.insert(word)
   
    def match_max_prefix(self, sentence):
        current_node = self.root
        longest_word = sentence[0]
        current_prefix = ""
        for ch in sentence:
            if ch in current_node:
                current_prefix += ch
                current_node = current_node[ch]
                if self.word_end in current_node:
                    longest_word = current_prefix
            else:
                break
        return longest_word
    def split_to_words(self, sentence, strategy="MAX_PREFIX"):
        rest = sentence
        words = []
        while len(rest) > 0:
            word = self.match_max_prefix(rest)
            words.append(word)
            if len(word) >= len(rest):
                break
            rest = rest[len(word):]
        return words
    
def test_trie():
    search_tree = trie()
    search_tree.insert("喜欢")
    search_tree.insert("非常")
    print(search_tree.root)
    print(search_tree.split_to_words("我非常喜欢你"))

if __name__ == "__main__":
    test_trie() 
