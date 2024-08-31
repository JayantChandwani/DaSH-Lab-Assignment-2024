#include <bits/stdc++.h>
using namespace std;

using ll = long long;

void solve(){
    ll N; cin >> N;
    ll ans{};
    for(ll i = 0; i < N; i++){
        ans += ((N/i))*i;
    }
    cout << ans << endl;
}

int main(){
    solve();
    return 0;
}