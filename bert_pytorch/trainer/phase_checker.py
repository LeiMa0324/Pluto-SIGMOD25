import os

class PhaseChecker:
    def __init__(self, method='vanilla',use_warmup_model = True,
                 warmup_model_dir = None,
                 warmup_epochs= 0, selection_epochs =0,
                 selection_gap = 0, retrain_epochs = 0,
                 first_freeze_epochs = 0):
        self.method = method
        self.use_warmup_model = use_warmup_model
        self.warmup_model_dir = warmup_model_dir
        self.phase_dict = {}
        self.selection_method_dict = {}
        self.epochs = 0

        self.warmup_epochs = warmup_epochs
        self.selection_epochs = selection_epochs
        # bgl 10
        self.first_freeze_epochs = first_freeze_epochs
        self.selection_gap = selection_gap
        self.retrain_epochs = retrain_epochs

        self.phase_define()

    def get_phase(self, epoch):
        return self.phase_dict[epoch]

    def phase_define(self):
        if self.method in ['vanilla', 'coteaching', 'coteaching_norm', 'logbert']:
            self.phase_dict = {e: "warm-up" for e in range(0, self.warmup_epochs )}
            self.epochs = self.warmup_epochs
        else:
            # try to load warm-up model
            if self.use_warmup_model:
                if self.warmup_model_dir is None or not os.path.exists(self.warmup_model_dir):
                    print(f"Invalid warm up model dir: {self.warmup_model_dir} ")
                    raise Exception(f"Invalid warm up model dir: {self.warmup_model_dir} ")

            else:
                self.phase_dict = {e: "warm-up" for e in range(0, self.warmup_epochs)}

            if self.method in ['ITLM', 'ITLM_norm', 'pluto']:

                selection_phase = {e: "selection" if e % self.selection_gap == 0 and
                                                     (
                                                                 e == self.warmup_epochs or e >= self.warmup_epochs + self.first_freeze_epochs)
                else "retrain"
                                   for e in range(self.warmup_epochs, self.warmup_epochs + self.selection_epochs)}
                self.phase_dict.update(selection_phase)
                retrain_phase = {e: "retrain" for e in range(self.warmup_epochs + self.selection_epochs,
                                                             self.warmup_epochs + self.selection_epochs + self.retrain_epochs)}
                self.phase_dict.update(retrain_phase)
                self.selection_method_dict.update({e: self.method for e in range(self.warmup_epochs,
                                                                                 self.warmup_epochs + self.selection_epochs + self.retrain_epochs)})
            elif self.method in ['fine+']:
                selection_phase = {e: "selection" if e % self.selection_gap == 0 else "freeze-selection"
                                   for e in range(self.warmup_epochs, self.warmup_epochs + self.selection_epochs)}
                self.phase_dict.update(selection_phase)
                self.selection_method_dict.update({e: self.method for e in range(self.warmup_epochs, self.warmup_epochs+self.selection_epochs+self.retrain_epochs)})

            else:
                raise NotImplementedError

        self.epochs = len(self.phase_dict.keys())
        return self.phase_dict

    def is_selection_phase(self, e):
        return  'selection' in self.phase_dict[e]

    def is_last_selection(self,e):
        if self.is_selection_phase(e):
            for epoch in range(e+1, self.epochs):
                is_selection = self.is_selection_phase(epoch)
                if  is_selection:
                    return False
            return True
        else:
            return False


    def selection_switching_to_retrain(self, e):
        return self.is_selection_phase(e) and e+1 < self.epochs and self.phase_dict[e+1] in ['retrain', 'recursive_select']

    def switching_selection_method(self, e):
        return self.selection_method_dict[e] != self.selection_method_dict[e+1] if e+1 < self.epochs else False

    def warmup_switched_to_selection(self, e):
        return self.is_selection_phase(e) and self.phase_dict[e-1]=='warm-up'

    def first_selection_phase(self, e):
        if e == 0:
            return True
        else:
            return self.warmup_switched_to_selection(e)

    def if_predict(self, epoch):
        if self.method in [ 'vanilla', 'coteaching', 'coteaching_norm', 'logbert']:
            return epoch % 5 ==0 or epoch ==self.epochs-1

        elif self.method in ['pluto',]:
            if ( epoch % 5 ==0) or epoch ==self.epochs-1:
                return True

        elif self.method in ['fine+']:
            if ( epoch % 5 ==0)  or epoch ==self.epochs-1:
                return True
        elif self.method in ["ITLM", "ITLM_norm"]:
            return  (epoch%5 ==0 or epoch ==self.epochs-1)
        else:
            return False

