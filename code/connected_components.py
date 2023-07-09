import matplotlib.pyplot as plt
import networkx as nx

G = nx.petersen_graph()

plt.subplot(121)
nx.draw(G, with_labels=False, font_weight='bold')

plt.subplot(122)
nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=False, font_weight='bold')

options = {
    'node_color': 'blue',
    'node_size': 100,
    'width': 2
}

plt.subplot(221)
nx.draw_random(G, **options)

plt.subplot(222)
nx.draw_circular(G, **options)

plt.subplot(223)
nx.draw_spectral(G, **options)

plt.subplot(224)
nx.draw_shell(G, nlist=[range(5,10), range(5)], **options)

plt.show()
