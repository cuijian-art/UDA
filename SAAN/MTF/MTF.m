function  M=MTF(data)
 
            X = data;
            m = length(X);
             
            X = (X - min(X))/(max(X) - min(X));
             
            N = length(X);
             
            Q = 4;
             
            X_Q = ones(1,N);
            j = 0;
             
            k = ones(1,Q+1);
            for i = 2 : Q+1
                
                while( sum(X < j) < N * (i-1) / Q)
                    j = j + 0.0001;
                end
                
                k(i) = j;
                
                X_Q(find(X < k(i) & X > k(i-1))) = i-1;
            end
             
            sum_14 = 0;
            sum_13 = 0;
            sum_24 = 0;
            sum_12 = 0;
            sum_23 = 0;
            sum_34 = 0;
            sum_11 = 0;
            sum_22 = 0;
            sum_33 = 0;
            sum_44 = 0;
            sum_21 = 0;
            sum_32 = 0;
            sum_43 = 0;
            sum_31 = 0;
            sum_42 = 0;
            sum_41 = 0;
             
            for i = 1:N-1
                switch(X_Q(i) - X_Q(i+1))
                    case -3
                        sum_14 = sum_14 + 1;
                    case -2
                        switch(X_Q(i))
                            case 1
                                sum_13 = sum_13 + 1;
                            case 2
                                sum_24 = sum_24 +1;
                        end
                    case -1
                        switch(X_Q(i))
                            case 1
                                sum_12 = sum_12 + 1;
                            case 2
                                sum_23 = sum_23 + 1;
                            case 3
                                sum_34 = sum_34 + 1;
                        end
                    case 0
                        switch(X_Q(i))
                            case 1
                                sum_11 = sum_11 + 1;
                            case 2
                                sum_22 = sum_22 + 1;
                            case 3
                                sum_33 = sum_33 + 1;
                            case 4 
                                sum_44 = sum_44 + 1;
                        end
                    case 1
                        switch(X_Q(i))
                            case 2
                                sum_21 = sum_21 + 1;
                            case 3
                                sum_32 = sum_32 + 1;
                            case 4
                                sum_43 = sum_43 + 1;
                        end
                    case 2
                        switch(X_Q(i))
                            case 3
                                sum_31 = sum_31 + 1;
                            case 4
                                sum_42 = sum_42 + 1;
                        end
                    case 3
                        sum_41 = sum_41 + 1;
                end
            end
             
            W = [sum_11 sum_12 sum_13 sum_14;
                sum_21 sum_22 sum_23 sum_24;
                sum_31 sum_32 sum_33 sum_34;
                sum_41 sum_42 sum_43 sum_44];
            W = W./repmat(sum(W),[4,1]);
             
            M = zeros(N,N);
            for i = 1: N
                for j = 1:N
                    M(i,j) = W(X_Q(i),X_Q(j));
                end
            end
