class DistanceCalculator:
    def __init__(self):
        self.cache = dict()

    def dist(self, p1, p2):
        if (p1, p2) not in self.cache:
            d = p1.geom.distance(p2.geom)
            self.cache[(p1, p2)] = d
            self.cache[(p2, p1)] = d
            return d
        else:
            return self.cache[(p1, p2)]


def isRkNN(p, q, k, index, semi_r_q, positive_dict, negative_dict, knn_radius_dict, dist_cal):
    if p in positive_dict:
        return True
    if p in negative_dict:
        return False
    if dist_cal.dist(p, q) <= semi_r_q:
        return True
    for neighbor in p.neighbors:
        if neighbor in positive_dict:
            p_disc = positive_dict[neighbor]
            if dist_cal.dist(p, q) + dist_cal.dist(p, p_disc) <= knn_radius_dict[p_disc]:
                positive_dict[p] = p_disc
                return True
        if neighbor in negative_dict:
            p_disc = negative_dict[neighbor]
            if dist_cal.dist(p, q) - dist_cal.dist(p, p_disc) > knn_radius_dict[p_disc]:
                negative_dict[p] = p_disc
                return False
    r = kNNRadius(p, k, index)
    knn_radius_dict[p] = r
    if dist_cal.dist(p, q) <= r:
        positive_dict[p] = p
        return True
    else:
        negative_dict[p] = p
        return False


def may_be_boundary_point(p, q, k, index, semi_r_q, positive_dict, negative_dict, knn_radius_dict, dist_cal):
    for neighbor in p.neighbors:
        if isRkNN(neighbor, q, k, index, semi_r_q, positive_dict, negative_dict, knn_radius_dict, dist_cal):
            return True
        elif not share_negative_disc_point(p, q, neighbor, negative_dict, knn_radius_dict, dist_cal):
            return True
    return False


def share_negative_disc_point(p, q, other, negative_dict, knn_radius_dict, dist_cal):
    disc_p = negative_dict[p]
    disc_other = negative_dict[other]
    if disc_p is disc_other:
        return True
    if dist_cal.dist(p, q) - dist_cal.dist(p, disc_other) > knn_radius_dict[disc_other]:
        return True
    if dist_cal.dist(other, q) - dist_cal.dist(other, disc_p) > knn_radius_dict[disc_p]:
        return True
    return False


def kNNRadius(q, k, index):
    if k == 0:
        return 0
    knn = list(index.nearest(q, k))
    return knn[-1][1]


def MonoRkNN(q, k, index):
    r_q = kNNRadius(q, k, index)
    semi_r_q = r_q / 2
    dist_cal = DistanceCalculator()
    positive_dict = dict()
    negative_dict = dict()
    knn_radius_dict = dict()
    candidates = list()
    visited = {q}
    for neighbor in q.neighbors:
        candidates.append(neighbor)
        visited.add(neighbor)
    while len(candidates) > 0:
        p = candidates.pop()
        if isRkNN(p, q, k, index, semi_r_q, positive_dict, negative_dict, knn_radius_dict, dist_cal):
            yield p
            for neighbor in p.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    candidates.append(neighbor)
        elif may_be_boundary_point(p, q, k, index, semi_r_q, positive_dict, negative_dict, knn_radius_dict,
                                   dist_cal):
            for neighbor in p.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    candidates.append(neighbor)


def BiRkNN(q, k, facility_index, user_index):
    r_q = kNNRadius(q, k - 1, facility_index)
    semi_r_q = r_q / 2
    dist_cal = DistanceCalculator()
    positive_dict = dict()
    negative_dict = dict()
    knn_radius_dict = dict()
    nn, nn_dist = list(user_index.nearest(q))[0]
    candidates = [nn]
    visited = {nn}
    while len(candidates) > 0:
        p = candidates.pop()
        if isRkNN(p, q, k, facility_index, semi_r_q, positive_dict, negative_dict, knn_radius_dict, dist_cal):
            yield p
            for neighbor in p.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    candidates.append(neighbor)
        elif may_be_boundary_point(p, q, k, facility_index, semi_r_q, positive_dict, negative_dict, knn_radius_dict,
                                   dist_cal):
            for neighbor in p.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    candidates.append(neighbor)
