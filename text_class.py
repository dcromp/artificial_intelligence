def levenshteinDistance(s1, s2, threshold=0.751):
    """
    The Levenshtein Distance between two words, 
    minimum number of single-character edits (insertions, deletions or substitutions)
    Args:
        s1 (str): The first string
        s2 (str): The second string
    Returns:
        float: The ratio of simularity, 1.0 being the same word, 0.0 being no similar letters in the two words
    """
    s1 = s1.lower()
    s2 = s2.lower()
    m = len(s1)
    n = len(s2)
    lensum = float(m + n)
    d = []           
    for i in range(m+1):
        d.append([i])        
    del d[0][0]    
    for j in range(n+1):
        d[0].append(j)       
    for j in range(1,n+1):
        for i in range(1,m+1):
            if s1[i-1] == s2[j-1]:
                d[i].insert(j,d[i-1][j-1])           
            else:
                minimum = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+2)         
                d[i].insert(j, minimum)
    ldist = d[-1][-1]
    ratio = (lensum - ldist)/lensum
    if ratio < threshold:
        return 0.
    return ratio


def message_simularity(message, target_class, dist_func=levenshteinDistance):
    """
    Returns the maximum simularity of message to the target class
    Args:
        message (list[str,...]): A list of strings that comprise the message
        target_class (list[str,...]): A list of strings that comprise the target class
    Returns:
        float: The total unweighted simularity between the message and the class
    """
    total_simularity = 0
    for word in message:
        total_simularity += max([dist_func(x, word) for x in target_class])
    return total_simularity

if __name__ == "__main__":
    
    message = ['wow', 'and', 'wonder', 'city']

    # Classes to score our message agaisnt
    class_a = ['Wow', 'Amazing', 'Window', 'Wonderful']
    class_b = ['Work', 'Job', 'Class', 'Tough']
    class_c = ['Wok', 'Woh', 'Wonderland', 'won']
    class_d = ['Well', 'Shell', 'Ship', 'Boat']
    class_index = {0: 'class_a', 1: 'class_b', 2: 'class_c', 3: 'class_d'}
    classes = [class_a, class_b, class_c, class_d]
    
    # Calculate the simulairty score for each class
    class_scores = [message_simularity(message, target_class) for target_class in classes]

    # Normalise predictions
    pred = [i / sum(class_scores) for i in class_scores]

    print("This message belongs to class {} with a score of {}".format(
        class_index[pred.index(max(pred))],
        max(pred))
         )
