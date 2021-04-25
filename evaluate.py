import jsonlines
import sys
import os


from eval.hypernym import HypernymScore
from eval.shortest_path import ShortestPath
from eval.stat_metrics import AUC
from utils.conll import write_output_file, write_connected_components
from coval.coval.conll import reader
from coval.coval.eval import evaluator



def eval_coref(gold, system):
    allmetrics = [('mentions', evaluator.mentions), ('muc', evaluator.muc),
                  ('bcub', evaluator.b_cubed), ('ceafe', evaluator.ceafe),
                  ('lea', evaluator.lea)]

    NP_only = False
    remove_nested = False
    keep_singletons = False
    min_span = False

    conll = 0

    doc_coref_infos = reader.get_coref_infos(gold, system, NP_only, remove_nested, keep_singletons, min_span)
    scores = {}

    for name, metric in allmetrics:
        recall, precision, f1 = evaluator.evaluate_documents(doc_coref_infos, metric, beta=1)
        scores[name] = [recall, precision, f1]

        if name in ["muc", "bcub", "ceafe"]:
            conll += f1

    scores['conll'] = conll
    return scores



def get_coref_scores(gold, system):
    output_path = 'tmp'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    write_output_file(gold, output_path, 'gold')
    write_output_file(system, output_path, 'system')

    write_connected_components(gold, output_path, 'gold')
    write_connected_components(system, output_path, 'system')

    coref_scores = eval_coref('tmp/gold_simple.conll', 'tmp/system_simple.conll')
    connected_scores = eval_coref('tmp/gold_connected.conll', 'tmp/system_connected.conll')

    return coref_scores, connected_scores



def main():
    gold_path = sys.argv[1]
    sys_path = sys.argv[2]
    hard = sys.argv[3] if len(sys.argv) > 3 else None

    with jsonlines.open(gold_path, 'r') as f:
        gold = [line for line in f]

    with jsonlines.open(sys_path, 'r') as f:
        system = [line for line in f]

    if hard:
        gold = [topic for topic in gold if topic[hard] == True]
        system = [topic for topic in system if topic['id'] in [x['id'] for x in gold]]

    print(len(gold))

    coref, connected = get_coref_scores(gold, system)
    print('Coref metrics')
    for metric, scores in coref.items():
        if metric != 'conll':
            recall, precision, f1 = scores
            print(metric.ljust(10), 'Recall: %.2f' % (recall * 100),
                  ' Precision: %.2f' % (precision * 100),
                  ' F1: %.2f' % (f1 * 100))
    conll_f1 = coref['conll'] / 3 * 100
    print('CoNLL score: %.2f' % conll_f1)

    hypernyms = HypernymScore(gold, system)
    print('Hierarchy'.ljust(15), 'Recall: %.2f' % (hypernyms.micro_recall * 100),
          ' Precision: %.2f' % (hypernyms.micro_precision * 100),
          ' F1: %.2f' % (hypernyms.micro_f1 * 100))



    path_based_3 = ShortestPath(gold, system, directed=True, with_tn=False)
    print('Path Ratio'.ljust(15),
          'Micro: %.2f' % (path_based_3.micro_average * 100),
          'Macro: %.2f' % (path_based_3.macro_average * 100))


    auc = AUC(gold, system)
    print('AUROC'.ljust(15),
          '%.2f' % (auc.score * 100))



if __name__ == '__main__':
    main()
